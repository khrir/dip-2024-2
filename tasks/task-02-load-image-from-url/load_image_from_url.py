import argparse
import numpy as np
import cv2 as cv
from urllib.request import urlopen

def load_image_from_url(url, **kwargs):
    """
    Loads an image from an Internet URL with optional arguments for OpenCV's cv.imdecode.
    
    Parameters:
    - url (str): URL of the image.
    - **kwargs: Additional keyword arguments for cv.imdecode (e.g., flags=cv.IMREAD_GRAYSCALE).
    
    Returns:
    - image: Loaded image as a NumPy array.
    """
    
    ### START CODE HERE ###

    req = urlopen(url)

    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv.imdecode(arr, **kwargs)

    return image

    ### END CODE HERE ###

parser = argparse.ArgumentParser(description='Load an image from an Internet URL.')
parser.add_argument('--url', type=str, help='URL of the image.')
args = parser.parse_args()

img = load_image_from_url(url=args.url, flags=cv.IMREAD_GRAYSCALE)

cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()