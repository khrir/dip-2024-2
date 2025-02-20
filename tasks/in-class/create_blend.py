import numpy as np
import cv2

alpha_slider_max = 100
title_window = 'Linear Blend'

# Carrega as imagens
f = cv2.imread('./input/f.png')
g = cv2.imread('./input/g.png')

# Verifica se as imagens foram carregadas corretamente
if f is None:
    print("Erro: Imagem 'f' não foi carregada. Verifique o caminho do arquivo.")
    exit()
if g is None:
    print("Erro: Imagem 'g' não foi carregada. Verifique o caminho do arquivo.")
    exit()

# Redimensiona a imagem 'g' para o mesmo tamanho da imagem 'f'
g = cv2.resize(g, (f.shape[1], f.shape[0]))

def on_trackbar(val):
    alpha = val / alpha_slider_max
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(f, alpha, g, beta, 0.0)
    cv2.imshow(title_window, dst)

def main():
    alpha_slider = 0

    # Cria a janela onde a imagem será exibida
    cv2.namedWindow(title_window, cv2.WINDOW_AUTOSIZE)

    # Cria o trackbar na janela correta
    trackbar_name = 'Alpha x %d' % alpha_slider_max
    cv2.createTrackbar(trackbar_name, title_window, 0, alpha_slider_max, on_trackbar)

    # Chama a função on_trackbar para exibir a imagem inicial
    on_trackbar(alpha_slider)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()