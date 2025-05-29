

import cv2
import numpy as np
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_gradient_magnitude
from numba import jit, prange

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_gradient_magnitude

def propagator(M, N, wavelength, area1, area2, z):
    i = np.arange(M) - M / 2
    j = np.arange(N) - N / 2

    alpha = wavelength * i[:, None] / area1  # Reshape i for broadcasting over MxN grid
    beta = wavelength * j[None, :] / area2   # Reshape j for broadcasting over MxN grid

    condition_mask = (alpha**2 + beta**2) <= 1

    sqrt_term = np.sqrt(1 - alpha**2 - beta**2, where=condition_mask)

    p = np.zeros((M, N), dtype=complex)
    p[condition_mask] = np.exp(-2j * np.pi * z * sqrt_term[condition_mask] / wavelength)
    return p

def funcAutoFocusAMS(I):
    return np.sum(np.abs(I))
    
def funcAutoFocusDFS(I):
    FX, FY = np.gradient(np.abs(I))
    temp = FX**2 + FY**2
    return np.var(temp)

def funcAutoFocusGRA(I):
    FX, FY = np.gradient(np.abs(I))
    temp = FX**2 + FY**2
    return np.mean(temp)

def AF_loop(d, lambda_, pixL, z_start, z_end, z_step,r_type, S,z):
    #print(lambda_, pixL, z_start, z_end, z_step,r_type, S,z)
    M, N = d.shape
    areax = M * pixL
    areay = N * pixL
    reconstructionCR = np.zeros((M, N, S))
    localDFSA = np.zeros(S)

    for ii in range(1,S):
            z0 = z_start + ii * z_step
            #print(z0)
            if r_type=="Spherical Wave":
                l1 = z0*areax/z
                l2 = z0*areay/z
                z1 = z0*(z-z0)/z
            else:
                l1 = areax
                l2 = areay
                z1 = z0
            hz_ = propagator(M, N, lambda_, l1, l2, -z1)
            or_ = np.fft.ifft2(np.fft.ifftshift(np.multiply(np.fft.fftshift(np.fft.fft2(d)), hz_)))
            abs_or = np.abs(or_)
            localDFSA[ii] = funcAutoFocusDFS(abs_or)
            reconstructionCR[:, :, ii] = abs_or
        #print(abs_or)

    return reconstructionCR, localDFSA

import numpy as np

def fRI_STTVv2(d, pixL, z0, lambda_, flag_pos, mu, muTV, t, nIter, nFGP, flag_obj):
    M, N = d.shape
    oprev = d.copy()
    u = oprev
    sprev = 1
    for n in range(nIter):
        uTemp = u
        print(n)
        # TV regularization
        on = fFGP(uTemp, muTV, nFGP)
        # Soft thresholding
        if flag_pos:
            on = np.maximum(0, on - mu * t)
        else:
            on = np.where(on > 0, np.maximum(0, on - mu * t), np.minimum(0, on + mu * t))
        
        # Update for acceleration using previous step
        s = 0.5 * (1 + (1 + 4 * sprev**2)**0.5)
        u = on + (sprev - 1) * (on - oprev) / s
        sprev = s
        oprev = on
        
        # Ensure values stay within bounds [0, 1]
        on = np.clip(on, 0, 1)
    
    return on

def fFGP(In, lambda_, nIter):
    M, N = In.shape
    rk = np.zeros((M-1, N))
    sk = np.zeros((M, N-1))
    pk1 = np.zeros((M-1, N))
    qk1 = np.zeros((M, N-1))
    tk = 1
    b = In
    for k in range(nIter):
        # Dual optimization steps
        Lpq = fL(rk, sk)
        
        Pc = fPC(b - lambda_ * Lpq)
        
        pt, qt = fLT(Pc)
        
        pt = (1 / (8 * lambda_)) * pt + rk
        qt = (1 / (8 * lambda_)) * qt + sk
        pk, qk = fPp(pt, qt)
        
        tk1 = (1 + (1 + 4 * tk**2)**0.5) / 2
        rk = pk + ((tk - 1) / tk1) * (pk - pk1)
        sk = qk + ((tk - 1) / tk1) * (qk - qk1)
        tk = tk1
        pk1, qk1 = pk, qk
    
    Lpq = fL(pk, qk)
    Iest = fPC(b - lambda_ * Lpq)
    return Iest

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
    
    # Interior points
    norm = np.maximum(1, np.sqrt(pt[:, :-1]**2 + qt[:-1, :]**2))
    r[:, :-1] = pt[:, :-1] / norm
    s[:-1, :] = qt[:-1, :] / norm
    
    # Edge cases
    r[:, -1] = pt[:, -1] / np.maximum(1, np.abs(pt[:, -1]))
    s[-1, :] = qt[-1, :] / np.maximum(1, np.abs(qt[-1, :]))
    
    return r, s