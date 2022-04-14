from os import getcwd
import argparse

from easyocr import Reader
import cv2 as cv
from PIL import Image


cwd = getcwd()
screenshots_directory = f'{cwd}/screenshots/'
text_and_coordinates_results_directory = f'{cwd}/text_reader_results/text_and_coordinates/'
images_results_directory = f'{cwd}/text_reader_results/images/'
languages = ["en", "ru"]

def get_text(screenshot):
    image_rgb = cv.imread(f'{screenshots_directory}/{screenshot}')

    reader = Reader(languages, gpu=True)
    results = reader.readtext(image_rgb)

    for result in results:
        np_result_image = cv.rectangle(image_rgb, (int(result[0][0][0]), int(result[0][0][1])),
                             (int(result[0][2][0]), int(result[0][2][1])), (0, 255, 0))
    result_image = Image.fromarray(np_result_image)

    return result_image, results

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot_name", type=str, default="screenshot.jpg")

    parser.set_defaults(train=False)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    result_image, boxes = get_text(args.screenshot_name)

    result_image.save(f'{images_results_directory}{args.screenshot_name[:-4]}.jpg')

    with open(f'{text_and_coordinates_results_directory}{args.screenshot_name[:-4]}.txt', 'w') as f:
        for box in boxes:
            f.write(str(box[0]))
            f.write('\n')
            f.write(box[1])
            f.write('\n')