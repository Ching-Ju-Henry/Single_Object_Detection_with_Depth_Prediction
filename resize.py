import numpy as np
import os
import cv2

#resize each image to corresponse to output of FCRN 
filename = '/home/henry/rolo/YOLO_tensorflow/photo/TV4.jpg'
img_resized = cv2.imread(filename)
new = cv2.resize(img_resized,(160,128))
det = '/home/henry/rolo/YOLO_tensorflow/resize/TV4_resize.jpg'
cv2.imwrite(det,new)
