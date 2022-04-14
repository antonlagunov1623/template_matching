from os import getcwd
import argparse

import cv2 as cv
from PIL import Image


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
        if match[0].distance < 0.25 * match[1].distance:
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

    np_result_image = cv.drawMatchesKnn(template, kp1, image, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    result_image = Image.fromarray(np_result_image)

    key_points_list = []
    for match in good_matches:
        img_idx = match[0].trainIdx
        (x2, y2) = kp2[img_idx].pt
        key_points_list.append((int(x2), int(y2)))

    return result_image, key_points_list

if __name__ == '__main__':
    args = get_args()

    result_image, key_points_list = find_best_matches_coordinates(args.template_name, args.screenshot_name)

    result_image.save(f'{images_results_directory}{args.template_name[:-4]}_{args.screenshot_name[:-4]}.jpg')

    with open(f'{coordinates_results_directory}{args.template_name[:-4]}_{args.screenshot_name[:-4]}.txt', 'w') as f:
        for key_point in key_points_list:
            f.write(str(key_point))
            f.write('\n')
