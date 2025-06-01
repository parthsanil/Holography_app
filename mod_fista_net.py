import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn  # In case your model class inherits from nn.Module
import torch.nn.functional as F
import os
import requests

# Weakened TransformBlock: smaller channels, shallower depth, no dropout
class TransformBlock(nn.Module):
    def __init__(self, in_channels, out_channels=4, depth=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            in_ch = in_channels if i == 0 else out_channels
            out_ch = out_channels if i < depth - 1 else in_channels
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    *( [nn.ReLU(inplace=True)] if i < depth - 1 else [] )
                )
            )

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features


# Weakened ResidualTransformBlock: reduced channels, no dropout
class ResidualTransformBlock(nn.Module):
    def __init__(self, skip_channels, out_channels=4, final_out_channels=1):
        super().__init__()
        self.depth = len(skip_channels)
        self.layers = nn.ModuleList()
        for i in range(self.depth):
            in_ch = skip_channels[-1 - i] + (out_channels if i > 0 else final_out_channels)
            out_ch = out_channels if i < self.depth - 1 else final_out_channels
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    *( [nn.ReLU(inplace=True)] if i < self.depth - 1 else [] )
                )
            )

    def forward(self, skips):
        x = skips[-1]
        for i in reversed(range(self.depth)):
            x = torch.cat([x, skips[i]], dim=1)
            x = self.layers[self.depth - 1 - i](x)
        return x


# Learnable soft-threshold with clamping
class SoftThreshold(nn.Module):
    def __init__(self, initial_theta=0.01):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(initial_theta, dtype=torch.float32))

    def forward(self, x):
        theta = torch.clamp(self.theta, 0.0, 0.1)  # clamp for stability
        return torch.sign(x) * F.relu(torch.abs(x) - theta)


class ProximalBlock(nn.Module):
    def __init__(self, channels=16, depth=4):
        super().__init__()
        self.F = TransformBlock(in_channels=channels, out_channels=16, depth=depth)

        # Get channel shapes from dummy input
        dummy = torch.zeros(1, channels, 32, 32)
        with torch.no_grad():
            skip_feats = self.F(dummy)
        skip_channels = [f.shape[1] for f in skip_feats]

        self.F_tilde = ResidualTransformBlock(
            skip_channels=skip_channels,
            out_channels=4,
            final_out_channels=channels
        )

    def forward(self, x, theta):
        residual = x
        features = self.F(x)

        # Apply soft thresholding only on last feature
        features[-1] = torch.sign(features[-1]) * F.relu(torch.abs(features[-1]) - theta)
        
        x = self.F_tilde(features)
        return residual + x
        
class FISTA_net(nn.Module):
    def __init__(self, n_iter=1):
        super().__init__()
        self.n_iter = n_iter

        # Per-iteration learnable momentum and step-size
        # self.rho = nn.ParameterList([
        #     nn.Parameter(torch.tensor(0.2)) for _ in range(n_iter)
        # ])
        # self.mu = nn.ParameterList([
        #     nn.Parameter(torch.tensor(0.2)) for _ in range(n_iter)
        # ])
        self.theta = nn.ParameterList([
            nn.Parameter(torch.tensor(0.05)) for _ in range(n_iter)
        ])

        # Denoiser module
        self.denoiser = ProximalBlock(channels=1, depth=4)

    def forward(self, T):
        B, _, M, N = T.shape
        r_list = []

        T = T.squeeze(0).squeeze(0)
        T_min = T.min()
        T_max = T.max()
        T = (T - T_min) / (T_max - T_min + 1e-8)
        T_orig = T.clone()
        P_prev = torch.zeros_like(T)

        for i in range(self.n_iter):
            # Gradient descent step
            # mu_k = self.mu[i]
            # r = T - mu_k * (T - T_orig)
            mm = T.real.unsqueeze(0).unsqueeze(0)
            r_list.append(mm)

            # Proximal mapping (with soft thresholding)
            theta_k = self.theta[i]
            P_real = self.denoiser(mm, theta_k).squeeze(0).squeeze(0)

            # Normalize P
            P_min = P_real.min()
            P_max = P_real.max()
            P = (P_real - P_min) / (P_max - P_min + 1e-8)

            # Momentum update
            # rho_k = self.rho[i]
            # T = P + rho_k * (P - P_prev)
            # P_prev = P.clone()
            T=P.clone()
            # Normalize T
            T_min = T.min()
            T_max = T.max()
            T = (T - T_min) / (T_max - T_min + 1e-8)

        # Final output normalization and reshaping
        out = T.real.unsqueeze(0).unsqueeze(0)
        out_min = out.min()
        out_max = out.max()
        out = (out - out_min) / (out_max - out_min + 1e-8)
        return out, r_list
 
 
def download_model(model_url, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading model from {model_url}...")
        response = requests.get(model_url)
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print("Model already exists locally.")

        
def fista_netmodel(image_np,model_class, model_url, model_local_path="fista_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    download_model(model_url, model_local_path)
    # Ensure image is 2D grayscale
    if image_np.ndim == 3:
        image_np = image_np.squeeze()
    assert image_np.ndim == 2, "Input image must be a 2D grayscale array"

    # Scale and convert to uint8 if needed
    if image_np.dtype != np.uint8:
        image_np = np.clip(image_np, 0, 1)
        image_np = (image_np * 255).astype(np.uint8)

    # Convert to PIL image
    image_pil = Image.fromarray(image_np)

    # Transform to tensor
    transform = transforms.ToTensor()
    input_tensor = transform(image_pil).unsqueeze(0).to(device)  # Shape: (1, 1, H, W)

    # Load model
    model = model_class(n_iter=1).to(device)
    model.load_state_dict(torch.load(model_local_path, map_location=device))
    model.eval()

    # Inference
    with torch.no_grad():
        output_tensor, _ = model(input_tensor)
    print(output_tensor.squeeze(0).squeeze(0).shape)
    return output_tensor.squeeze(0).squeeze(0).cpu().numpy()  # Return as NumPy array (H, W)
