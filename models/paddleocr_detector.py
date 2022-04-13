from os import listdir, getcwd
import string
import time

from paddleocr import PaddleOCR, draw_ocr
from matplotlib import pyplot as plt
import cv2 as cv
from PIL import Image

cwd = getcwd()
screenshots_directory = f'{cwd}/screenshots/'
languages = ["en", "ru"]

for site in listdir(screenshots_directory):
    for screenshot in listdir(f'{screenshots_directory}{site}/'):
        start = time.time()
        ocr = PaddleOCR(use_angle_cls=True, lang='ru')
        result = ocr.ocr(f'{screenshots_directory}{site}/{screenshot}', cls=True)
        end = time.time()
        print(end - start)
        for line in result:
            print(line)

        image_rgb = cv.imread(f'{screenshots_directory}{site}/{screenshot}')
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        for i, bbox in enumerate(boxes):
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))

            # text = cleanup_text(text)
            cv.rectangle(image_rgb, tl, br, (0, 255, 0), 2)
            cv.putText(image_rgb, txts[i], (tl[0], tl[1] - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        plt.subplot(121), plt.imshow(image_rgb)
        plt.title('Matching result'), plt.xticks([]), plt.yticks([])

        plt.show()

        # break

    # break