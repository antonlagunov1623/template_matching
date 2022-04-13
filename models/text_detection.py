from os import listdir, getcwd

import pyocr
import pyocr.builders
import cv2 as cv
from PIL import Image

cwd = getcwd()
screenshots_directory = f'{cwd}/screenshots/'

for site in listdir(screenshots_directory):
    for screenshot in listdir(f'{screenshots_directory}{site}/'):

        tools = pyocr.get_available_tools()
        tool = tools[0]

        # line_and_word_boxes = tool.image_to_string(
        #     Image.open(f'{screenshots_directory}{site}/{screenshot}'), lang="rus",
        #     builder=pyocr.builders.LineBoxBuilder()
        # )

        txt = tool.image_to_string(
            Image.open(f'{screenshots_directory}{site}/{screenshot}'),
            lang='rus',
            builder=pyocr.builders.TextBuilder()
        )

        file = open(f'{cwd}/pyocr_results/{site}_{screenshot}.txt', 'w')
        file.write(txt)
        file.close()