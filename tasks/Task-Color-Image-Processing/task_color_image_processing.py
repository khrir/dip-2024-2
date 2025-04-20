import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load the images
    chips = cv2.imread("../../img/chips.png")
    flowers = cv2.imread("../../img/flowers.jpg")
    hsv_disk = cv2.imread("../../img/hsv_disk.png")
    lena = cv2.imread("../../img/lena.png")
    monkey = cv2.imread("../../img/monkey.jpeg")
    rgb = cv2.imread("../../img/rgb.png")
    rgbcube_kBKG = cv2.imread("../../img/rgbcube_kBKG.png")
    strawberries = cv2.imread("../../img/strawberries.tif")
    
    # converte as imagens pra RGB
    chips = cv2.cvtColor(chips, cv2.COLOR_BGR2RGB)
    flowers = cv2.cvtColor(flowers, cv2.COLOR_BGR2RGB)
    hsv_disk = cv2.cvtColor(hsv_disk, cv2.COLOR_BGR2RGB)
    lena = cv2.cvtColor(lena, cv2.COLOR_BGR2RGB)
    monkey = cv2.cvtColor(monkey, cv2.COLOR_BGR2RGB)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgbcube_kBKG = cv2.cvtColor(rgbcube_kBKG, cv2.COLOR_BGR2RGB)
    strawberries = cv2.cvtColor(strawberries, cv2.COLOR_BGR2RGB)

    # display_hist(chips, "Chips Image Histogram")
    # visualize_channels(chips)
    convert_and_display_color_spaces(chips)

##1. Display Color Histograms for RGB Images
# Objective: Calculate and display separate histograms for the R, G, and B channels of a color image.
# Topics: Color histograms, channel separation.
# Challenge: Compare histograms of different images (e.g., nature vs. synthetic images).
def display_hist(image: np.ndarray, title: str) -> None:
    # separa os canais de cor
    b_channel, g_channel, r_channel = cv2.split(image)
    
    # cria um histograma para cada canal
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(b_channel.ravel(), bins=256, color='blue', alpha=0.7)
    plt.title('Blue Channel Histogram')
    
    plt.subplot(1, 3, 2)
    plt.hist(g_channel.ravel(), bins=256, color='green', alpha=0.7)
    plt.title('Green Channel Histogram')
    
    plt.subplot(1, 3, 3)
    plt.hist(r_channel.ravel(), bins=256, color='red', alpha=0.7)
    plt.title('Red Channel Histogram')
    
    plt.suptitle(title)    
    plt.show()

# 2. Visualize Individual Color Channels
# Objective: Extract and display the Red, Green, and Blue channels of a color image as grayscale and pseudo-colored images.
# Topics: Channel separation and visualization.
# Bonus: Reconstruct the original image using the separated channels.
def visualize_channels(image: np.ndarray) -> None:
    # separa os canais de cor
    b_channel, g_channel, r_channel = cv2.split(image)
    
    # cria uma imagem em escala de cinza para cada canal
    b_gray = cv2.cvtColor(b_channel, cv2.COLOR_GRAY2BGR)
    g_gray = cv2.cvtColor(g_channel, cv2.COLOR_GRAY2BGR)
    r_gray = cv2.cvtColor(r_channel, cv2.COLOR_GRAY2BGR)
    
    # cria uma imagem pseudo-colorida para cada canal
    b_pseudo = cv2.applyColorMap(b_channel, cv2.COLORMAP_JET)
    g_pseudo = cv2.applyColorMap(g_channel, cv2.COLORMAP_JET)
    r_pseudo = cv2.applyColorMap(r_channel, cv2.COLORMAP_JET)
    
    # exibe as imagens
    plt.figure(figsize=(15, 10))    
    plt.subplot(2, 3, 1)
    plt.imshow(b_gray)
    plt.title('Blue Channel (Grayscale)')
    
    plt.subplot(2, 3, 3)
    plt.imshow(g_gray)
    plt.title('Green Channel (Grayscale)')
    
    plt.subplot(2, 3, 5)
    plt.imshow(r_gray)
    plt.title('Red Channel (Grayscale)')
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(b_pseudo)
    plt.title('Blue Channel (Pseudo-colored)')
    
    plt.subplot(2, 3, 3)
    plt.imshow(g_pseudo)
    plt.title('Green Channel (Pseudo-colored)')
    
    plt.subplot(2, 3, 5)
    plt.imshow(r_pseudo)
    plt.title('Red Channel (Pseudo-colored)')
    
    # reconstrua a imagem original
    merged_image = cv2.merge((b_channel, g_channel, r_channel))
    plt.figure(figsize=(15, 10))

    plt.subplot(1, 1, 1)
    plt.imshow(merged_image)
    plt.title('Reconstructed Image')
    
    plt.show()

# 3. Convert Between Color Spaces (RGB ↔ HSV, LAB, YCrCb, CMYK)
# Objective: Convert an RGB image to other color spaces and display the result.
# Topics: Color space conversion.
# Challenge: Display individual channels from each converted space
def convert_and_display_color_spaces(image: np.ndarray) -> None:
    # converte a imagem para HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv_image)
    
    # converte a imagem para LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # converte a imagem para YCrCb
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)
    
    # converte a imagem para CMYK
    c_channel, m_channel, y2_channel, k_channel = rgb_to_cmyk(image)

    # exibe as imagens
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(hsv_image)
    plt.title('HSV Image')
    
    plt.subplot(2, 3, 3)
    plt.imshow(lab_image)
    plt.title('LAB Image')
    
    plt.subplot(2, 3, 5)
    plt.imshow(ycrcb_image)
    plt.title('YCrCb Image')

    # plt.subplot(2, 3, 1)
    # plt.imshow(cmyk_image)
    # plt.title('CMYK Image')
    
    plt.subplot(2, 3, 2)
    plt.imshow(h_channel, cmap='gray')
    plt.title('H Channel (HSV)')
    
    plt.subplot(2, 3, 4)
    plt.imshow(l_channel, cmap='gray')
    plt.title('L Channel (LAB)')
    
    plt.subplot(2, 3, 2)
    plt.imshow(y_channel, cmap='gray')
    plt.title('Y Channel (YCrCb)')

    plt.subplot(2, 3, 3)
    plt.imshow(c_channel, cmap='gray')
    plt.title('C Channel (CMYK)')
    
    plt.show()

def rgb_to_cmyk(image: np.ndarray) -> np.ndarray:
    CMYK_SCALE = 100
    RGB_SCALE = 255

    r, g, b = cv2.split(image)

    if (r == 0) and (g == 0) and (b == 0):
        # black
        return 0, 0, 0, CMYK_SCALE

    # rgb [0,255] -> cmy [0,1]
    c = 1 - r / RGB_SCALE
    y = 1 - b / RGB_SCALE

    # extract out k [0, 1]
    min_cmy = min(c, m, y)
    c = (c - min_cmy) / (1 - min_cmy)
    m = (m - min_cmy) / (1 - min_cmy)
    y = (y - min_cmy) / (1 - min_cmy)    
    m = 1 - g / RGB_SCALE

    k = min_cmy

    # rescale to the range [0,CMYK_SCALE]
    return c * CMYK_SCALE, m * CMYK_SCALE, y * CMYK_SCALE, k * CMYK_SCALE

if __name__ == "__main__":
    main()