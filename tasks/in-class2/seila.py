import numpy as np
import matplotlib.pyplot as plt
import cv2

cv2.namedWindow('Img1', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('Img2', cv2.WINDOW_KEEPRATIO)

ksizex = 0
ksizey = 0

cv2.createTrackbar('ksizex', 'BarX', 0, 63, lambda x: x)
cv2.createTrackbar('ksizey', 'BarY', 0, 63, lambda x: x)

