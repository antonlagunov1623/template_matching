# Screenshot processing

### Template matching
##### Build
~~~
python models/template_matching.py --screenshot_name={screenshot_name}.{format} --template_name={template_name}.{format}
~~~
##### Input
Templates images should be store in 'template' folder, screenshots should be store in 'screenshots' folder
##### Output
After processing script saves image with found key points as a result in 'template_matching_results/images/{template_name}_{screenshot_name}.jpg'. 
Coordinates of key points are in 'template_matching_results/images/{template_name}_{screenshot_name}.txt'

### Text reader
##### Build
~~~
python models/text_reader.py --screenshot_name={screenshot_name}.{format}
~~~
##### Input
Screenshots should be store in 'screenshots' folder
##### Output
After processing script saves image with bounded boxes as a result in 'text_reader_results/images/{screenshot_name}.jpg'. 
Coordinates of boxes corners and text are storing in 'text_reader_results/text_and_coordinates/{screenshot_name}.txt'
##### Format of coordinates file
~~~
[[left_bottom_corner_x_1, left_bottom_corner_y_1], [left_top_corner_x_1, left_top_corner_y_1], [right_top_corner_x_1, right_top_corner_y_1], [right_bottom_corner_x_1, right_bottom_corner_y_1]]
text_1
[[left_bottom_corner_x_2, left_bottom_corner_y_2], [left_top_corner_x_2, left_top_corner_y_2], [right_top_corner_x_2, right_top_corner_y_2], [right_bottom_corner_x_2, right_bottom_corner_y_2]]
text_2
...
~~~