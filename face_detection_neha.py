import cv2
import numpy as np

# load the cascade
# create the CascadeClassifier Object
cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# read the input image
image = cv2.imread('legend.jpg')

# reads images in the form of numpyarray
print(type(image))
print(image.shape)

# convert into grayscale or reading the image as gray scale image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect files ( search the co-ordinates of the image)
detect_face_coordinate = cascade_face.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=4)

#show the coordinates of the rectangle
print(detect_face_coordinate)

# draw rectangle around the faces
for(a, b, w, h) in detect_face_coordinate:
    cv2.rectangle(image, (a, b), (a+w, b+h), (179, 245, 66), 2)

# display the output
cv2.imshow('legend',image)

# wait till any key is pressed
cv2.waitKey(0)

cv2.destroyAllWindows()

