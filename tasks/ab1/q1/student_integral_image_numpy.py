import numpy as np


def compute_integral_image(image: np.ndarray) -> np.ndarray:
    """
    Computes the integral image using NumPy.


    Parameters:
        image (np.ndarray): 2D grayscale image.


    Returns:
        np.ndarray: Integral image.
    """
    # TODO: Implement your solution here
    integral_image = np.zeros_like(image, dtype=np.float64)
    rows, cols = image.shape
    
    for i in range(rows):
        for j in range(cols):
            if i == 0 and j == 0:
                integral_image[i, j] = image[i, j]
            elif i == 0:
                integral_image[i, j] = integral_image[i, j - 1] + image[i, j]
            elif j == 0:
                integral_image[i, j] = integral_image[i - 1, j] + image[i, j]
            else:
                integral_image[i, j] = (integral_image[i - 1, j] +
                                        integral_image[i, j - 1] -
                                        integral_image[i - 1, j - 1] +
                                        image[i, j])
    return integral_image


# Define test image
image = np.array([
    [0.32285394, 0.95322289, 0.31806831],
    [0.12936134, 0.45275244, 0.60094833],
    [0.71811803, 0.49059312, 0.38843348],
], dtype=np.float64)


expected_result = np.array([
    [0.32285394, 1.27607683, 1.59414514],
    [0.45221528, 1.85819061, 2.77720725],
    [1.17033331, 3.06690176, 4.37435188],
], dtype=np.float64)


integral_image = compute_integral_image(image)


if (expected_result == integral_image).all():
    print(f'Success!')


print(f'Original image: \n{image}\n')
print(f'Expected result: \n{expected_result}\n')
print(f'Your result: \n{integral_image}\n')


# Explique como a imagem integral é usada para acelerar o algoritmo de detecção
# de objetos com Haar Cascades, como no método Viola-Jones. 
# Por que ela é fundamental para viabilizar a execução em tempo real?

# deu tempo não :/