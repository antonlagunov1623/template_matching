from os import listdir, getcwd

import cv2 as cv
from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
import imutils
from PIL import Image


cwd = getcwd()
screenshots_directory = f'{cwd}/screenshots/'
elements_directory = f'{cwd}/elements/'
coordinates_results_directory = f'{cwd}/template_matching_results/coordinates/'
images_results_directory = f'{cwd}/template_matching_results/images/'

def find_matches(template, image):
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(image, None)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=15)
    # search_params = dict(checks=50)  # or pass empty dictionary

    # flann = cv.FlannBasedMatcher(index_params, search_params)
    #
    # matches = flann.knnMatch(des1, des2, k=2)

    good = []
    good_matches = []
    for i, match in enumerate(matches):
        if match[0].distance < 0.25 * match[1].distance:
            good.append([match[0]])
            good_matches.append(matches[i])
    return kp1, kp2, good, good_matches

for site in listdir(screenshots_directory):
    for element in listdir(f'{elements_directory}{site}/'):
        template = cv.imread(f'{elements_directory}{site}/{element}', 0)
        template_rgb = cv.imread(f'{elements_directory}{site}/{element}', cv.IMREAD_COLOR)
        for screenshot in listdir(f'{screenshots_directory}{site}/'):
            image = cv.imread(f'{screenshots_directory}{site}/{screenshot}', 0)
            image_rgb = cv.imread(f'{screenshots_directory}{site}/{screenshot}', cv.IMREAD_COLOR)

            # h, w = template_rgb.shape[:2]
            # method = cv.TM_CCOEFF_NORMED
            # threshold = 0.5
            # res = cv.matchTemplate(image_rgb, template_rgb, method)
            #
            # max_val = 1
            # while max_val > threshold:
            #     min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            #     if max_val > threshold:
            #         res[max_loc[1] - h // 2:max_loc[1] + h // 2 + 1, max_loc[0] - w // 2:max_loc[0] + w // 2 + 1] = 0
            #         image = cv.rectangle(image, (max_loc[0], max_loc[1]), (max_loc[0] + w + 1, max_loc[1] + h + 1),
            #                               (0, 255, 0))
            #
            # plt.imshow(image), plt.show()

            kp1_1, kp2_1, good_1, good_matches_1 = find_matches(template, image)
            kp1_2, kp2_2, good_2, good_matches_2 = find_matches(cv.bitwise_not(template), image)
            kp1 = None
            kp2 = None
            good = None
            good_matches = None
            if len(good_1) == 0 and len(good_2) == 0:
                kp1 = kp1_1
                kp2 = kp2_1
                good = good_1
                good_matches = good_matches_1
            elif len(good_1) == 0 and len(good_2) != 0:
                kp1 = kp1_2
                kp2 = kp2_2
                good = good_2
                good_matches = good_matches_2
            else:
                kp1 = kp1_1
                kp2 = kp2_1
                good = good_1
                good_matches = good_matches_1

            img3 = cv.drawMatchesKnn(template, kp1, image, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            im = Image.fromarray(img3)
            im.save(f'{images_results_directory}{element[:-4]}_{screenshot[:-4]}.jpeg')

            list_kp1 = []
            list_kp2 = []

            for mat in good_matches:
                img1_idx = mat[0].queryIdx
                img2_idx = mat[0].trainIdx

                (x1, y1) = kp1[img1_idx].pt
                (x2, y2) = kp2[img2_idx].pt

                list_kp1.append((int(x1), int(y1)))
                list_kp2.append((int(x2), int(y2)))

            with open(f'{coordinates_results_directory}{element[:-4]}_{screenshot[:-4]}.txt', 'w') as f:
                for coor in list_kp2:
                    f.write(str(coor))
                    f.write('\n')
