from os import getcwd
import argparse

import cv2 as cv
import numpy as np
import imutils
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

def find_similar(key_points, tepmlate_size, image_size):
    good = []
    for i, kp in enumerate(key_points):
        good_group = []
        xS = 0
        xE = 0
        yS = 0
        yE = 0
        if kp[0] - tepmlate_size[0] < 0:
            xS = 0
        else:
            xS = kp[0] - tepmlate_size[0]
        if kp[0] + tepmlate_size[0] > image_size[0]:
            xE = image_size[0]
        else:
            xE = kp[0] + tepmlate_size[0]
        if kp[1] - tepmlate_size[1] < 0:
            yS = 0
        else:
            yS = kp[1] - tepmlate_size[1]
        if kp[1] + tepmlate_size[1] > image_size[1]:
            yE = image_size[1]
        else:
            yE = kp[1] + tepmlate_size[1]
        good_group.append(kp)
        for j, p in enumerate(key_points[1:]):
            if p[0] > xS and p[0] < xE and p[1] > yS and p[1] < yE:
                good_group.append(p)
        if len(good_group) >= 5:
            good.append(good_group)
    return good

def find_matches(template_name, screenshot_name):
    template = cv.imread(f'{templates_directory}/{template_name}', 0)
    image = cv.imread(f'{screenshots_directory}/{screenshot_name}', 0)

    sift = cv.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(template, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image, None)
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    key_points_list = []
    for match in matches:
        img_idx = match.trainIdx
        (x2, y2) = keypoints_2[img_idx].pt
        key_points_list.append((int(x2), int(y2)))

    key_points_list = find_similar(key_points_list, template.shape, image.shape)
    # for group in key_points_list:
    #     for point in group:
    #         plt.plot(point[0], point[1], color='red', marker='v')
    # plt.imshow(image)
    # plt.show()

    return key_points_list

def template_matching(template_name, screenshot_name):
    bboxes = []
    gray_bboxes = []
    temp = None
    template = cv.imread(f'{templates_directory}/{template_name}')
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    template = cv.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]

    while True:
        found = None
        image = cv.imread(f'{screenshots_directory}/{screenshot_name}')
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        for box in bboxes:
            for i in range(box[0][0], box[1][0]):
                for j in range(box[0][1], box[1][1]):
                    gray[j][i] = 0
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break
            edged = cv.Canny(resized, 50, 200)
            result = cv.matchTemplate(edged, template, cv.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)
        if found[0] > 0.5:
            (_, maxLoc, r) = found
            (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
            (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
            # cv.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            # plt.imshow(image), plt.show()
            gray_bboxes.append(((maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH)))
            bboxes.append(((startX, startY), (endX, endY)))
        else:
            break
        if temp != None and temp - found[0] > 0.15:
            bboxes.pop()
            break
        temp = found[0]
    return bboxes

if __name__ == '__main__':
    args = get_args()
    key_points_list = find_matches(args.template_name, args.screenshot_name)
    bboxes = template_matching(args.template_name, args.screenshot_name)

    if len(key_points_list) > 0 and len(bboxes) == 0:
        data = image.imread(f'{screenshots_directory}/{args.screenshot_name}')
        for group in key_points_list:
            for point in group:
                plt.plot(point[0], point[1], color='red', marker='v')
                plt.imshow(data)
                plt.savefig(f'{images_results_directory}{args.template_name[:-4]}_{args.screenshot_name[:-4]}.jpg')
        with open(f'{coordinates_results_directory}{args.template_name[:-4]}_{args.screenshot_name[:-4]}.txt',
                  'w') as f:
            for group in key_points_list:
                for kp in group:
                    f.write(str(kp))
                    f.write('\n')
    elif len(key_points_list) >= 0 and len(bboxes) > 0:
        data = image.imread(f'{screenshots_directory}/{args.screenshot_name}')
        for bb in bboxes:
            figure = cv.rectangle(data, bb[0], bb[1], (255, 0, 0), 2)
        plt.imshow(data)
        plt.savefig(f'{images_results_directory}{args.template_name[:-4]}_{args.screenshot_name[:-4]}.jpg')
        coordinates = []
        for bb in bboxes:
            coordinates.append((int((bb[0][0] + bb[1][0]) / 2), int((bb[0][1] + bb[1][1]) / 2)))
        with open(f'{coordinates_results_directory}{args.template_name[:-4]}_{args.screenshot_name[:-4]}.txt',
                  'w') as f:
            for kp in coordinates:
                f.write(str(kp))
                f.write('\n')
    else:
        data = image.imread(f'{screenshots_directory}/{args.screenshot_name}')
        plt.imshow(data)
        plt.savefig(f'{images_results_directory}{args.template_name[:-4]}_{args.screenshot_name[:-4]}.jpg')
        with open(f'{coordinates_results_directory}{args.template_name[:-4]}_{args.screenshot_name[:-4]}.txt',
                  'w') as f:
            f.write('\n')