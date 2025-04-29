import cv2
import numpy as np

def verify_laplacian_isotropy(image_path, angle=45):
    """
    Aplica rotação + Laplaciano e Laplaciano + rotação na imagem de entrada
    e retorna o coeficiente de correlação de Pearson entre os dois resultados.

    Parâmetros:
        image_path (str): Caminho para a imagem em tons de cinza.
        angle (float): Ângulo de rotação em graus.

    Retorno:
        float: Coeficiente de correlação de Pearson entre as duas imagens.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to load.")


    # TODO: Implementar solução aqui

    #aplicar rotação + Laplaciano
    rotated_image = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotated_image, (image.shape[1], image.shape[0]))
    result_1 = cv2.Laplacian(rotated_image, cv2.CV_64F)

    #aplicar Laplaciano + rotação
    laplacian_image = cv2.Laplacian(image, cv2.CV_64F)
    rotated_laplacian_image = cv2.getRotationMatrix2D((laplacian_image.shape[1] / 2, laplacian_image.shape[0] / 2), angle, 1)
    result_2 = cv2.warpAffine(laplacian_image, rotated_laplacian_image, (laplacian_image.shape[1], laplacian_image.shape[0]))

    # Flatten and compute correlation
    corr = np.corrcoef(result_1.flatten(), result_2.flatten())[0, 1]
    print(f'Correlation coefficient: {corr}')
    return corr


verify_laplacian_isotropy('example_image.png', angle=45)
# Expected: Correlation coefficient: 0.9621492663463063


verify_laplacian_isotropy('checkerboard_image.png', angle=45)
# Expected: Correlation coefficient: 0.9383739474511829
