import cv2
import numpy as np

class ImageConh:
    def __init__(self):
        self.f = cv2.imread('./input/f.png')
        self.g = cv2.imread('./input/g.png')
        self.g = cv2.resize(self.g, (self.f.shape[1], self.f.shape[0]))

    def on_trackbar(self, val):
        alpha = val / 100
        beta = (1.0 - alpha)
        dst = cv2.addWeighted(self.f, alpha, self.g, beta, 0.0)
        cv2.imshow('H', dst)

    def main(self):    
        # create windows with sliders a and b
        cv2.namedWindow('F', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('G', cv2.WINDOW_AUTOSIZE)

        # create trackbars for a and b
        cv2.createTrackbar('a', 'F', 0, 100, lambda x: None)
        cv2.createTrackbar('b', 'G', 0, 100, lambda x: None)

        # Show h = a * F + b * G
        while True:
            a = cv2.getTrackbarPos('a', 'F') / 100
            b = cv2.getTrackbarPos('b', 'G') / 100

            h = cv2.addWeighted(self.f, a, self.g, b, 0)

            cv2.imshow('H', h)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ic = ImageConh()
    ic.main()

