import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from skimage.io import imread
import Reconstruction
import imageio.v3 as iio
import time



def fHz(rows, cols, lambda_, areax, areay, z):
    y = np.arange(1, rows + 1)
    x = np.arange(1, cols + 1)
    X, Y = np.meshgrid(x, y)
    
    alpha = lambda_ * (X - (cols / 2) - 1) / areax
    beta = lambda_ * (Y - (rows / 2) - 1) / areay
    
    p = np.zeros((rows, cols), dtype=complex)
    mask = (alpha**2 + beta**2) <= 1
    p[mask] = np.exp(2 * np.pi * 1j * z * np.sqrt(1 - alpha[mask]**2 - beta[mask]**2) / lambda_)
    
    return p
def funcAutoFocusDFS(I):
    FX, FY = np.gradient(np.abs(I))
    temp = FX**2 + FY**2
    return np.var(temp)
def fSFPRNL(d, pixL, z0, lambda_, muTV, nIter, nFGP, gamma, mu1):
    M, N = d.shape
    areax = N * pixL
    areay = M * pixL
    print(pixL, z0, lambda_, muTV, nIter, nFGP, gamma, mu1)
    hz = fHz(M, N, lambda_, areax, areay, z0)
    T = ifft2(ifftshift(fftshift(fft2(d)) * hz))
    Q1 = 0.25
    P1 = 0

    for _ in range(nIter):
        #print(_)
        t1 = ifft2(ifftshift(fftshift(fft2(T)) * hz))
        t2 = np.sqrt(d) * (t1 / np.abs(t1))
        g = ifft2(ifftshift(fftshift(fft2(t2)) * np.conj(hz)))

        gx, gy = np.gradient(g)
        norm_Grad_g = np.sqrt(gx**2 + gy**2)
        gx /= norm_Grad_g
        gy /= norm_Grad_g

        gxx, _ = np.gradient(gx)
        _, gyy = np.gradient(gy)
        divG = gxx + gyy

        S = (mu1 / 2) * (g + (gamma / 2) * divG)
        P = fFGP(np.real(S), muTV, nFGP) + 1j * fFGP(np.imag(S), muTV, nFGP)
        Q = 0.5 * (1 + np.sqrt(1 + 4 * Q1**2))

        T = P + (Q1 - 1) * (P - P1) / Q
        P1 = P
        Q1 = Q

    return T


def fFGP(In, lambda_, nIter):
    M, N = In.shape
    rk = np.zeros((M-1, N))
    sk = np.zeros((M, N-1))
    pk1 = np.zeros((M-1, N))
    qk1 = np.zeros((M, N-1))

    tk = 1
    b = In

    for _ in range(nIter):
        Lpq = fL(rk, sk)
        Pc = fPC(b - lambda_ * Lpq)
        pt, qt = fLT(Pc)

        pt = (1 / (8 * lambda_)) * pt
        qt = (1 / (8 * lambda_)) * qt

        pt += rk
        qt += sk

        pk, qk = fPp(pt, qt)

        tk1 = (1 + np.sqrt(1 + 4 * tk**2)) / 2
        rk = pk + ((tk - 1) / tk1) * (pk - pk1)
        sk = qk + ((tk - 1) / tk1) * (qk - qk1)

        tk = tk1
        pk1 = pk
        qk1 = qk

    Lpq = fL(pk, qk)
    return fPC(b - lambda_ * Lpq)


def fL(p, q):
    M, N = p.shape[0] + 1, q.shape[1] + 1
    Lpq = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            if i == 0 and j == 0:
                Lpq[i, j] = p[i, j] + q[i, j]
            elif i == 0 and j == N - 1:
                Lpq[i, j] = p[i, j] - q[i, j - 1]
            elif i == M - 1 and j == 0:
                Lpq[i, j] = q[i, j] - p[i - 1, j]
            elif i == 0 and 0 < j < N - 1:
                Lpq[i, j] = p[i, j] + q[i, j] - q[i, j - 1]
            elif 0 < i < M - 1 and j == 0:
                Lpq[i, j] = p[i, j] + q[i, j] - p[i - 1, j]
            elif i == M - 1 and j < N - 1:
                Lpq[i, j] = q[i, j] - p[i - 1, j] - q[i, j - 1]
            elif i < M - 1 and j == N - 1:
                Lpq[i, j] = p[i, j] - p[i - 1, j] - q[i, j - 1]
            elif i == M - 1 and j == N - 1:
                Lpq[i, j] = -p[i - 1, j] - q[i, j - 1]
            else:
                Lpq[i, j] = p[i, j] + q[i, j] - p[i - 1, j] - q[i, j - 1]
    return Lpq


def fLT(x):
    p = x[:-1, :] - x[1:, :]
    q = x[:, :-1] - x[:, 1:]
    return p, q


def fPC(temp):
    return np.clip(temp, -1, 1)


def fPp(pt, qt):
    M, N = pt.shape[0] + 1, qt.shape[1] + 1
    r = np.zeros((M - 1, N))
    s = np.zeros((M, N - 1))

    denom = np.maximum(1, np.sqrt(pt[:, :-1]**2 + qt[:-1, :]**2))
    r[:, :-1] = pt[:, :-1] / denom
    r[:, -1] = pt[:, -1] / np.maximum(1, np.abs(pt[:, -1]))

    denom = np.maximum(1, np.sqrt(pt[:, :-1]**2 + qt[:-1, :]**2))
    s[:-1, :] = qt[:-1, :] / denom
    s[-1, :] = qt[-1, :] / np.maximum(1, np.abs(qt[-1, :]))

    return r, s


def reconnew(image_path):
    H = iio.imread(image_path).astype(np.float64)

    # If you need grayscale and the image is RGB, convert it manually
    if H.ndim == 3:
        # Convert RGB to grayscale using luminosity method
        H = 0.2989 * H[..., 0] + 0.5870 * H[..., 1] + 0.1140 * H[..., 2]
    H = H / np.max(H) 
    pixL = 4.59e-6  # camera pixel size
    lambda_ = 660e-9  # wavelength of light source
    M, N = H.shape
    #z0 = 22e-3
    areax = N * pixL  # area side length in meter
    areay = M * pixL  # area side length in meter
    #print(H, lambda_, pixL, 0, 30* 1e-3, 0.2* 1e-3, '', 150, 1)
    reconstructionCR, localDFSA = Reconstruction.AF_loop(H, lambda_, pixL, 1*1e-3, 100* 1e-3, 0.1* 1e-3, '', 1000, 1)
    z0 = np.argmax(localDFSA)
    #print(z0)
    z0=1e-3+(z0*0.1* 1e-3)
    print(z0)
    hz = fHz(M, N, lambda_, areax, areay, -z0)
    d = ifft2(ifftshift(fftshift(fft2(H)) * hz))
    d = np.abs(d) ** 2
    d = d / np.mean(d) 
    muTV = 0.1
    nIter = 15
    nFGP = 3
    gamma = 0.1
    mu1 = 0.1
    
    start_time = time.time()
    Tn = fSFPRNL(H, pixL, z0, lambda_, muTV, nIter, nFGP, gamma, mu1)
    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.4f} seconds")
    
    
    # Process amplitude
    amplitude = (np.abs(Tn) - np.min(np.abs(Tn))) / (np.max(np.abs(Tn)) - np.min(np.abs(Tn)))
    amplitude = (amplitude * 255).astype(np.uint8)  # Scale to 8-bit for proper saving
    
    # Process phase
    phase = (np.angle(Tn) - np.min(np.angle(Tn))) / (np.max(np.angle(Tn)) - np.min(np.angle(Tn)))
    
    phase = (phase * 255).astype(np.uint8)  # Scale to 8-bit for proper saving 
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Amplitude")
    plt.imshow(amplitude, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Phase")
    plt.imshow(phase, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    plt.show()
    return amplitude,phase

if __name__ == '__main__':
    reconnew(r"D:\PRoject\MTP\Examples\Test_1.tif")
###fSFPRNL  Execution time: 4.0225 seconds
### --------CNNN_Denoise   FLOPs: 1.43 GMac  1.1ms
#####FISTA NET -------      3.22GMACs      6ms 9.81k params
#########FISTA   16.35 seconds



#\section{Architectural Comparison}
#\begin{table}[h!]
#    \centering
#    \caption{Comparison of FISTA-Net and Modified FISTA-Net Architectures}
#    \resizebox{\textwidth}{!}{%
#    \begin{tabular}{@{}lcc@{}}
#        \toprule
#        \textbf{Component} & \textbf{FISTA-Net} & \textbf{Modified FISTA-Net} \\
#        \midrule
#        Iterations & 5  & Single pass \\
#        Gradient Descent Step & Learnable step size & Learnable step size \\
#        CNN1 (Feature Extraction) & 5 layers, 32 filters, \(3 \times 3\) & 3 layers, 16 filters, \(3 \times 3\) \\
#        CNN2 (Reconstruction) & 5 layers, 32 filters, \(3 \times 3\) & 2 layers, 1 filter, \(3 \times 3\) \\
#        Residual Learning and Skip Connections & No & Yes  \\
#        Decoder Style & Plain CNN & U-Net style \\
#        SSIM in Loss & No & Yes \\
#        Total Parameters & 38,017 & 6,588  \\
#        \bottomrule
#    \end{tabular}
#    }
#    \label{tab:fista_comparison}
#\end{table}



#\noindent The modified model achieves better performance despite fewer parameters by leveraging residual learning and perceptual training objectives.
