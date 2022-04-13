from os import listdir, getcwd
import time

from easyocr import Reader
from matplotlib import pyplot as plt
import cv2 as cv

cwd = getcwd()
screenshots_directory = f'{cwd}/screenshots/'
languages = ["en", "ru"]

def cleanup_text(text):
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

for site in listdir(screenshots_directory):
    for screenshot in listdir(f'{screenshots_directory}{site}/'):
        # print(site)
        if site == '1screeshots':
            image_rgb = cv.imread(f'{screenshots_directory}{site}/{screenshot}')

            start = time.time()
            reader = Reader(languages, gpu=True, cudnn_benchmark=True)
            results = reader.readtext(image_rgb)
            end = time.time()
            print(end - start)

            # for (bbox, text, prob) in results:
            #     (tl, tr, br, bl) = bbox
            #     tl = (int(tl[0]), int(tl[1]))
            #     tr = (int(tr[0]), int(tr[1]))
            #     br = (int(br[0]), int(br[1]))
            #     bl = (int(bl[0]), int(bl[1]))
            #
            #     # text = cleanup_text(text)
            #     cv.rectangle(image_rgb, tl, br, (0, 255, 0), 2)
            #     cv.putText(image_rgb, text, (tl[0], tl[1] - 10),
            #                 cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            #
            #
            #
            # plt.subplot(121), plt.imshow(image_rgb)
            # plt.title('Matching result'), plt.xticks([]), plt.yticks([])
            #
            # print(results)
            #
            # plt.show()

        # break

    # break

            # cv.imshow("Image", image_rgb)
            # cv.waitKey(0)