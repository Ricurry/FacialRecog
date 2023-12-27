# Take everything from set1 and resize it to 255x255
# Save it to set2

import cv2 as cv
import numpy as np
import os
import glob

# Get all the png files from set1
# Resize them to 255x255

# Get all the png files from set1
path = 'notset'
files = glob.glob(path + '/*.png')
i = 0
for file in files:
    img = cv.imread(file)
    resized_image = cv.resize(img, (255, 255))
    cv.imwrite('notset2/%d.png' % i, resized_image)
    i+=1
    cv.waitKey(50)
    print('Resizing image %d' % i)
print('Done resizing images')
