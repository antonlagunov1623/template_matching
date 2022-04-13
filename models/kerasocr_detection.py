from os import listdir, getcwd
import string

import keras_ocr
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np

cwd = getcwd()
screenshots_directory = f'{cwd}/screenshots/'
languages = ["en", "ru"]
# glagolitsa = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"

detector = keras_ocr.detection.Detector(weights='clovaai_general', load_from_torch=False,
                                        optimizer='adam', backbone_name='vgg')
recognizer = keras_ocr.recognition.Recognizer(alphabet=None, weights='kurapan', build_params=None)

for site in listdir(screenshots_directory):
    for screenshot in listdir(f'{screenshots_directory}{site}/'):
        image_rgb = cv.imread(f'{screenshots_directory}{site}/{screenshot}')

        detection_result = detector.detect([image_rgb], detection_threshold=0.7, text_threshold=0.4,
                                           link_threshold=0.4, size_threshold=10)

        recognizer_result = recognizer.recognize_from_boxes([image_rgb], detection_result)

        # plt.subplot(121), plt.imshow(image_rgb)
        # plt.title('Matching result'), plt.xticks([]), plt.yticks([])

        print(detection_result)
        print(recognizer_result)
        print(detection_result.shape)

        for i, bbox in enumerate(detection_result):
            # (tl, tr, br, bl) = bbox
            tl = (np.int32(bbox[0][0]), np.int32(bbox[0][1]))
            tr = (np.int32(bbox[1][0]), np.int32(bbox[1][1]))
            br = (np.int32(bbox[2][0]), np.int32(bbox[2][1]))
            bl = (np.int32(bbox[3][0]), np.int32(bbox[3][1]))

            # text = cleanup_text(text)
            print(tl)
            print(br)
            cv.rectangle(image_rgb, tl, br, (0, 255, 0), 2)
            cv.putText(image_rgb, recognizer_result[i], (tl[0], tl[1] - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        plt.subplot(121), plt.imshow(image_rgb)
        plt.title('Matching result'), plt.xticks([]), plt.yticks([])

        plt.show()

        # plt.show()

        # break

    # break