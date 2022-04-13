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

        k_size = 3
        kernel = np.ones((k_size, k_size), np.uint8)
        image_dilated = cv.dilate(image_rgb, kernel, iterations=1)

        image_eroded = cv.erode(image_rgb, kernel, iterations=1)

        # image_blurred = cv.GaussianBlur(image_eroded, (0, 0), 3)

        # image_gray = cv.cvtColor(image_rgb, cv.COLOR_BGR2GRAY)
        # image = cv.medianBlur(image_gray, 5)
        #
        result = pytesseract.image_to_string(image_eroded, lang='rus')

        # h, w, c = image_rgb.shape
        # boxes = pytesseract.image_to_boxes(image_rgb)
        # for b in boxes.splitlines():
        #     b = b.split(' ')
        #     img = cv.rectangle(image_rgb, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

        # cv.imshow('img', image_rgb)

        file = open(f'{cwd}/tesseract_results/{site}_{screenshot}.txt', 'w')
        file.write(result)
        file.close()

        # plt.subplot(121), plt.imshow(image_rgb)
        # plt.title('Matching result'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122), plt.imshow(image_rgb)
        # plt.title('Template'), plt.xticks([]), plt.yticks([])
        # plt.suptitle(meth)

        plt.show()