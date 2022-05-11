from os import getcwd
import argparse
import math

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image


cwd = getcwd()
screenshots_directory = f'{cwd}/screenshots/'
templates_directory = f'{cwd}/templates/'
coordinates_results_directory = f'{cwd}/template_matching_results/coordinates/'
images_results_directory = f'{cwd}/template_matching_results/images/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot_name", type=str, default="screenshot.jpg")
    parser.add_argument("--template_name", type=str, default="template.jpg")

    parser.set_defaults(train=False)

    return parser.parse_args()

def find_matches(template, image):
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(image, None)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    good_matches = []
    for i, match in enumerate(matches):
        if match[0].distance < 0.5 * match[1].distance:
            good.append([match[0]])
            good_matches.append(matches[i])

    return kp1, kp2, good, good_matches

def find_best_matches_coordinates(template_name, screenshot_name):
    template = cv.imread(f'{templates_directory}/{template_name}', 0)
    image = cv.imread(f'{screenshots_directory}/{screenshot_name}', 0)

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

    # np_result_image = cv.drawMatchesKnn(template, kp1, image, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # result_image = Image.fromarray(np_result_image)

    key_points_list = []
    for match in good_matches:
        img_idx = match[0].trainIdx
        (x2, y2) = kp2[img_idx].pt
        key_points_list.append((int(x2), int(y2)))

    return key_points_list

def template_matching(template_name, screenshot_name):
    main_img = cv.imread(f'{screenshots_directory}/{screenshot_name}')
    gray_img = cv.cvtColor(main_img, cv.COLOR_BGR2GRAY)

    template = cv.imread(f'{templates_directory}/{template_name}', 0)
    template_edged = cv.Canny(template, 50, 200)
    ht_t, wd_t = template_edged.shape
    found = None

    for scale in np.linspace(0.1, 1, 10)[::-1]:
        resized = cv.resize(gray_img, dsize=(0, 0), fx=scale, fy=scale)
        r = gray_img.shape[1] / float(resized.shape[1])

        if resized.shape[0] < ht_t or resized.shape[1] < wd_t:
            break
        edged = cv.Canny(resized, 50, 200)
        result = cv.matchTemplate(edged, template_edged, cv.TM_CCOEFF_NORMED)
        (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(result)
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + wd_t) * r), int((maxLoc[1] + ht_t) * r))
    # figure = cv.rectangle(main_img, (startX, startY), (endX, endY), (255, 0, 0), 2)
    # plt.imshow(main_img), plt.show()

    return (startX, startY), (endX, endY)


if __name__ == '__main__':
    args = get_args()
    # data = image.imread(f'{screenshots_directory}/{args.screenshot_name}')
    # data = cv.imread(f'{screenshots_directory}/{args.screenshot_name}')
    key_points_list = find_best_matches_coordinates(args.template_name, args.screenshot_name)
    (startX, startY), (endX, endY) = template_matching(args.template_name, args.screenshot_name)
    print(key_points_list)

    if len(key_points_list) > 0:
        clear_list = []
        for key_point in key_points_list:
            if (key_point[0] >= startX and key_point[0] <= endX) and (key_point[1] >= startY and key_point[1] <= endY):
                clear_list.append(key_point)
        if len(clear_list) == 0 and len(key_points_list) < 10:
            data = image.imread(f'{screenshots_directory}/{args.screenshot_name}')
            figure = cv.rectangle(data, (startX, startY), (endX, endY), (255, 0, 0), 2)
            plt.imshow(data)
            plt.savefig(f'{images_results_directory}{args.template_name[:-4]}_{args.screenshot_name[:-4]}.jpg')
            coordinate = (int((startX + endX) / 2), int((startY + endY) / 2))
            with open(f'{coordinates_results_directory}{args.template_name[:-4]}_{args.screenshot_name[:-4]}.txt',
                      'w') as f:
                f.write(str(coordinate))
                f.write('\n')
        else:
            data = image.imread(f'{screenshots_directory}/{args.screenshot_name}')
            for point in clear_list:
                plt.plot(point[0], point[1], color='red', marker='v')
            plt.imshow(data)
            plt.savefig(f'{images_results_directory}{args.template_name[:-4]}_{args.screenshot_name[:-4]}.jpg')

            with open(f'{coordinates_results_directory}{args.template_name[:-4]}_{args.screenshot_name[:-4]}.txt',
                      'w') as f:
                for key_point in clear_list:
                    f.write(str(key_point))
                    f.write('\n')
    else:
        data = image.imread(f'{screenshots_directory}/{args.screenshot_name}')
        figure = cv.rectangle(data, (startX, startY), (endX, endY), (255, 0, 0), 2)
        plt.imshow(data)
        plt.savefig(f'{images_results_directory}{args.template_name[:-4]}_{args.screenshot_name[:-4]}.jpg')
        coordinate = (int((startX+endX)/2), int((startY+endY)/2))
        with open(f'{coordinates_results_directory}{args.template_name[:-4]}_{args.screenshot_name[:-4]}.txt',
                  'w') as f:
            f.write(str(coordinate))
            f.write('\n')
