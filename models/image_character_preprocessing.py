from os import listdir, getcwd

import pytesseract
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

cwd = getcwd()
screenshots_directory = f'{cwd}/screenshots/'

for site in listdir(screenshots_directory):
    for screenshot in listdir(f'{screenshots_directory}{site}/'):
        image_rgb = cv.imread(f'{screenshots_directory}{site}/{screenshot}')

        plt.subplot(121), plt.imshow(image_rgb)
        plt.title('RGB image'), plt.xticks([]), plt.yticks([])
        plt.show()

        k_size=3
        kernel = np.ones((k_size, k_size), np.uint8)
        image_dilated = cv.dilate(image_rgb, kernel, iterations=1)

        plt.subplot(121), plt.imshow(image_dilated)
        plt.title('Dilated image'), plt.xticks([]), plt.yticks([])
        plt.show()

        image_eroded = cv.erode(image_rgb, kernel, iterations=1)

        plt.subplot(121), plt.imshow(image_eroded)
        plt.title('Eroded image'), plt.xticks([]), plt.yticks([])
        plt.show()

        image_blurred = cv.GaussianBlur(image_eroded, (0, 0), 3)

        plt.subplot(121), plt.imshow(image_eroded)
        plt.title('Blurred image'), plt.xticks([]), plt.yticks([])
        plt.show()

        break