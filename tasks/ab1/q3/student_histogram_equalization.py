
import cv2
import numpy as np

def equalize_histogram(image_path: str) -> np.ndarray:
    """
    Realiza equalização de histograma apenas no canal Y de uma imagem RGB convertida para YCrCb.

    Parâmetros:
        image_path (str): Caminho para a imagem RGB.

    Retorno:
        np.ndarray: Imagem RGB com o canal Y equalizado.
    """
    # TODO: Implemente sua solução aqui
    # converte a imagem para YCrCb
    image = cv2.imread(image_path)

    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)

    # aplica a equalização de histograma no canal Y
    y_eq = cv2.equalizeHist(y_channel)
    ycrcb_eq = cv2.merge((y_eq, cr_channel, cb_channel))

    equalized_image_ycrcb = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    return equalized_image_ycrcb


student_result = equalize_histogram('unequal_lighting_color_image.png')
expected_result = cv2.imread('expected_result.png')

if (expected_result == student_result).all():
    print('Success!')