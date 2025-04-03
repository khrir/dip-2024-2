# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    # Your implementation here
    return {
        "mse": calculate_mse(i1, i2),
        "psnr": calculate_psnr(i1, i2),
        "ssim": calculate_ssim(i1, i2),
        "npcc": calculate_npcc(i1, i2)
    }

# MSE é a média dos erros quadráticos entre os pixels de duas imagens.
def calculate_mse(i1: np.ndarray, i2: np.ndarray) -> float:
    mse = np.mean((i1 - i2) ** 2, dtype=np.float64)

    return mse

# PSNR é a medição de quanto ruído afeta a qualidade de um sinal.
# É frequentemente usado para avaliar a qualidade de imagens e vídeos
def calculate_psnr(i1: np.ndarray, i2: np.ndarray) -> float:
    mse = calculate_mse(i1, i2)
    psnr = 10 * np.log10(1 / mse)

    return psnr

# SSIM é uma métrica que considera a percepção humana da imagem na comparação de similaridade.
# Ela considera mudanças de luminância, contraste e estrutura entre duas imagens.
def calculate_ssim(i1: np.ndarray, i2: np.ndarray) -> float:
    mu1 = np.mean(i1)
    mu2 = np.mean(i2)

    sigma1_sq = np.var(i1)
    sigma2_sq = np.var(i2)
    sigma12 = np.cov(i1.flatten(), i2.flatten())[0, 1]

    c1 = 6.5025
    c2 = 58.5225

    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))

    return ssim

# ρ (rho) (X, Y) =  cov(X, Y) / (std(X) * std(Y))
# cov(X, Y) = E[(X - σX)(Y - σY)]
# std(X) = sqrt(E[(X - μX)^2]), onde μX é a média de X
# (NPCC) é uma medida estatística que quantifica a relação linear entre duas variáveis.
def calculate_npcc(i1: np.ndarray, i2: np.ndarray) -> float:
    i1_flat = i1.flatten()
    i2_flat = i2.flatten()

    mean_i1 = np.mean(i1_flat)
    mean_i2 = np.mean(i2_flat)

    numerator = np.sum((i1_flat - mean_i1) * (i2_flat - mean_i2))
    denominator = np.sqrt(np.sum((i1_flat - mean_i1) ** 2) * np.sum((i2_flat - mean_i2) ** 2))

    npcc = numerator / denominator

    return npcc

if __name__ == "__main__":
    # Teste com duas imagens de exemplo
    i1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    i2 = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]], dtype=np.float32)

    result = compare_images(i1, i2)
    print(result)