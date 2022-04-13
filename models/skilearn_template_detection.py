from os import listdir, getcwd

import numpy as np
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import match_template
from skimage.feature import peak_local_max
import cv2 as cv
import statistics


cwd = getcwd()
screenshots_directory = f'{cwd}/screenshots/'
elements_directory = f'{cwd}/elements/'

for site in listdir(screenshots_directory):
    for element in listdir(f'{elements_directory}{site}/'):
        template_rgb = cv.imread(f'{elements_directory}{site}/{element}', 0)
        # template_gray = cv.cvtColor(template_rgb, cv.COLOR_BGR2GRAY)
        for screenshot in listdir(f'{screenshots_directory}{site}/'):
            image_rgb = cv.imread(f'{screenshots_directory}{site}/{screenshot}')
            image_gray = cv.cvtColor(image_rgb, cv.COLOR_BGR2GRAY)

            resulting_image = match_template(image_gray, template_rgb)
            # x, y = np.unravel_index(np.argmax(resulting_image), resulting_image.shape)
            # template_width, template_height = template_rgb.shape
            # rect = plt.Rectangle((y, x), template_height, template_width,
            #                      color='r', fc='none')
            # plt.figure(num=None, figsize=(8, 6), dpi=80)
            # plt.gca().add_patch(rect)
            # imshow(resulting_image)

            # template_width, template_height = template_rgb.shape
            # plt.figure(num=None, figsize=(8, 6), dpi=80)
            # for x, y in peak_local_max(resulting_image, threshold_abs=0.7,
            #                            exclude_border=20):
            #     rect = plt.Rectangle((y, x), template_height, template_width,
            #                          color='r', fc='none')
            #     plt.gca().add_patch(rect)
            # imshow(image_gray)

            template_width, template_height = template_rgb.shape
            matched_list = []
            for x, y in peak_local_max(resulting_image, threshold_abs=0.50, exclude_border=10):
                rect = plt.Rectangle((y, x), template_height, template_width)
                coord = Rectangle.get_bbox(rect).get_points()
                matched_list.append(coord)

            matched_patches = [image_gray[int(match[0][1]):int(match[1][1]),
                               int(match[0][0]):int(match[1][0])] for match in matched_list]
            difference = [abs(i.flatten() - template_rgb.flatten()) for i in matched_patches]
            summed_diff = [array.sum() for array in difference]
            final_patches = list(zip(matched_list, summed_diff))
            statistics.mean(np.float32(summed_diff))

            summed_diff = np.array(summed_diff)
            filtered_list_mean = list(filter(lambda x: x[1] <=
                                                       summed_diff.mean(), final_patches))
            filtered_list_median = list(filter(lambda x: x[1] <=
                                                         np.percentile(summed_diff, 50),
                                               final_patches))
            filtered_list_75 = list(filter(lambda x: x[1] <=
                                                     np.percentile(summed_diff, 75),
                                           final_patches))

            fig, ax = plt.subplots(1, 3, figsize=(17, 10), dpi=80)
            template_width, template_height = template_rgb.shape
            for box in filtered_list_mean:
                patch = Rectangle((box[0][0][0], box[0][0][1]), template_height,
                                  template_width, edgecolor='b',
                                  facecolor='none', linewidth=3.0)
                ax[0].add_patch(patch)
            ax[0].imshow(image_rgb, cmap='gray');
            ax[0].set_axis_off()
            for box in filtered_list_median:
                patch = Rectangle((box[0][0][0], box[0][0][1]), template_height,
                                  template_width, edgecolor='b',
                                  facecolor='none', linewidth=3.0)
                ax[1].add_patch(patch)
            ax[1].imshow(image_rgb, cmap='gray');
            ax[1].set_axis_off()
            for box in filtered_list_75:
                patch = Rectangle((box[0][0][0], box[0][0][1]), template_height,
                                  template_width,
                                  edgecolor='b', facecolor='none',
                                  linewidth=3.0)
                ax[2].add_patch(patch)
            ax[2].imshow(image_rgb, cmap='gray');
            ax[2].set_axis_off()
            fig.tight_layout()

            plt.show()

            print(resulting_image)

            # break

        # break

    # break