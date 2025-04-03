# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:
    # Your implementation here
    return {
        "translated": get_translated(img),
        "rotated": get_rotated(img),
        "stretched": get_stretched(img),
        "mirrored": get_mirrored(img),
        "distorted": get_distorted(img)
    }

# desloca a imagem para a direita e para baixo
def get_translated(img: np.ndarray) -> np.ndarray:
    translated_img = np.zeros_like(img)
    translated_img[1:, 1:] = img[:-1, :-1]
    return translated_img

# gira a imagem 90 graus no sentido horário
def get_rotated(img: np.ndarray) -> np.ndarray: 
    rotated_img = np.rot90(img, axes=(-2,-1))
    return rotated_img

# estica a imagem horizontalmente
def get_stretched(img: np.ndarray) -> np.ndarray:
    stretched_img = np.zeros((img.shape[0], int(img.shape[1] * 1.5)), dtype=img.dtype)
    stretched_img[:, :img.shape[1]] = img
    return stretched_img

# espelha a imagem horizontalmente
def get_mirrored(img: np.ndarray) -> np.ndarray:
    mirrored_img = np.flip(img, axis=1)
    return mirrored_img

# aplica uma distorção de barril na imagem
def get_distorted(img: np.ndarray) -> np.ndarray:
    distorted_img = np.zeros_like(img)
    h, w = img.shape
    center_x, center_y = w // 2, h // 2

    for y in range(h):
        for x in range(w):
            dx = x - center_x
            dy = y - center_y
            r = np.sqrt(dx**2 + dy**2)
            if r > 0:
                factor = 1 - (r / max(center_x, center_y))**2
                new_x = int(center_x + dx * factor)
                new_y = int(center_y + dy * factor)
                if 0 <= new_x < w and 0 <= new_y < h:
                    distorted_img[y, x] = img[new_y, new_x]
    return distorted_img

if __name__ == "__main__":
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
    transformations = apply_geometric_transformations(img)
    for name, transformed_img in transformations.items():
        print(f"{name}:")
        print(transformed_img)