import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_image(seed, width, height, mean, std):
    """
    Generates a grayscale image with pixel values sampled from a normal distribution.

    Args:
        seed (int): Random seed for reproducibility (student's registration number).
        width (int): Width of the generated image.
        height (int): Height of the generated image.
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.

    Returns:
        image (numpy.ndarray): The generated image.
    """
    ### START CODE HERE ###
    np.random.seed(seed)

    # Generate pixel values from a normal distribution
    pixel_values = np.random.normal(mean, std, (height, width))

    # Clip pixel values to the valid range [0, 255] for grayscale images
    pixel_values = np.clip(pixel_values, 0, 255)

    # Convert pixel values to 8-bit unsigned integers (required for image representation)
    image = pixel_values.astype(np.uint8)    

    ### END CODE HERE ###

    return image

def generate_histogram_opencv2(image):
    # Calculate the histogram of the image
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    hist_height = 256
    hist_width = 256

    # Normalize the histogram to fit within the plot dimensions
    hist = cv2.normalize(hist, hist, 0, hist_height, cv2.NORM_MINMAX)

    # Create a blank image to draw the histogram
    hist_image = np.zeros((hist_height, hist_width), dtype=np.uint8)

    # Draw the histogram bars
    for i in range(256):
        cv2.line(hist_image, (i, hist_height), (i, hist_height - int(hist[i].item())), 255, 1)
    
    
    # Display the histogram
    cv2.imwrite('./output/hist_cv2.png', cv2.cvtColor(hist_image, cv2.COLOR_GRAY2BGR))

def generate_histogram_plt(image):
    # Flatten the image into a 1D array of pixel values
    pixel_values = image.flatten()

    # Plot the histogram
    plt.hist(pixel_values, bins=256, range=(0, 256), color='black', alpha=0.75)
    plt.title("Histogram of Pixel Values")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])
    plt.grid(axis='y', alpha=0.75)

    # Save the histogram plot
    plt.savefig('./output/hist_plt.png')
    plt.close()

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Generate an image with pixel values sampled from a normal distribution.")

    parser.add_argument('--registration_number', type=int, required=True, help="Student's registration number (used as seed)")
    parser.add_argument('--width', type=int, required=True, help="Width of the image")
    parser.add_argument('--height', type=int, required=True, help="Height of the image")
    parser.add_argument('--mean', type=float, required=True, help="Mean of the normal distribution")
    parser.add_argument('--std', type=float, required=True, help="Standard deviation of the normal distribution")
    parser.add_argument('--output', type=str, required=True, help="Path to save the generated image")

    args = parser.parse_args()

    # Generate the image
    image = generate_image(args.registration_number, args.width, args.height, args.mean, args.std)

    # Save the generated image
    cv2.imwrite(args.output, image)

    print(f"Image successfully generated and saved to {args.output}")

    # Generate the histogram of the image
    generate_histogram_plt(image)
    generate_histogram_opencv2(image)

if __name__ == "__main__":
    main()