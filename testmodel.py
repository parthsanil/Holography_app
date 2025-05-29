import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class index to label mapping (update with your actual mapping if needed)
idx_to_class = {
    0: 'Marine Snow', 1: 'Thick Elongated Diatom', 2: 'Fecal Pellets', 3: 'Unknown Diatom',
    4: 'Protoperidinium', 5: 'Diatom 4 (c. concavicornus)', 6: 'Diatom 3 (ditylum sp.)',
    7: 'Square Coscinodiscus', 8: 'Unknown-Eastsound', 9: 'Ceratium fusus (singular)',
    10: 'Decoy', 11: 'Phaeocystis sp', 12: 'Thalassionema nitzschoides', 13: 'Appendicularian',
    14: 'Fuzzy Diatom', 15: 'Diatom 2', 16: 'Chaetoceros socialis', 17: 'Ceratium furca (singular)',
    18: 'Sea Urchin Larvae', 19: 'Ceratium furca (dividing)', 20: 'Copepod Nauplii',
    21: 'Ceratium muelleri (singular)', 22: 'C-Shaped Diatom', 23: 'Thalassiosira sp',
    24: 'Tadpole-Stage Larvae', 25: 'Elongated Ciliate', 26: 'Ceratium muelleri (dividing)',
    27: 'Ciliate', 28: 'Ceratium fusus (dividing)', 29: 'Diatom 1 (c. debilis)',
    30: 'Box Chain Diatom', 31: 'Thin Elongated Diatom', 32: 'Copepod',
    33: 'Round Coscinodiscus', 34: 'Karenia brevis', 35: 'Aggregate',
}
# Image preprocessing (same as training

class ResizeWithPadding:
    def __init__(self, size=224, fill=255):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray((img * 255).astype(np.uint8))  # scale to 0–255 if float
        if img.mode != 'RGB':
            img = img.convert('RGB')
        ratio = min(self.size / img.width, self.size / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img_resized = img.resize(new_size, Image.BILINEAR)
        new_img = Image.new("RGB", (self.size, self.size), (self.fill, self.fill, self.fill))
        paste_coords = ((self.size - new_size[0]) // 2, (self.size - new_size[1]) // 2)
        new_img.paste(img_resized, paste_coords)
        return new_img

# 🔹 Preprocessing (same as training)
transform = transforms.Compose([
    ResizeWithPadding(224, fill=255),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
])

# Load model
def load_model(path="/kaggle/input/model_frspnl/other/default/1/best_model_fSFPRNL.pth", num_classes=36):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(2048, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Predict function
def predict_image(image, model):
    # Convert grayscale to RGB
    if image.ndim == 2:
        image = np.stack([image]*3, axis=-1)
    elif image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    # Ensure image is uint8 and scaled properly
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 1)  # Assumes float in [0,1]
        image = (image * 255).astype(np.uint8)

    # Debug save before transformation
    debug_img = Image.fromarray(image)
    debug_img.save("transformed_debug.png")
    print("✅ Saved transformed image as 'transformed_debug.png'")

    # Apply transforms
    image = transform(debug_img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)

    class_idx = predicted.item()
    class_name = idx_to_class.get(class_idx, str(class_idx))
    print(f"✅ Predicted Class: {class_name} (index {class_idx})")
    return class_idx, class_name

# Usage example
#model = load_model(r"D:\PRoject\Re_ Matlab code\Resnet_model\best_model_fSFPRNL.pth", num_classes=36)
###image_path = r"D:\PRoject\MTP\Triples_36_Classes\Triples_36_Classes - Copy\Triples_Raw-Subtracted-Splat\Fecal Pellets\amplitude\S04a_002904_00001_subtracted.png"  # Replace with your image path
#predict_image(image_path, model)
