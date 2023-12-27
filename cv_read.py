import cv2 as cv
import numpy as np
import pyvirtualcam
cam = cv.VideoCapture(0)
#Set camera resolution to 256x256
cam.set(3, 255)
cam.set(4, 255)
#Import pb file 'frozen_graph.pb' to opencv
net = cv.dnn.readNetFromTensorflow('frozen_graph.pb')
print("OpenCV model was successfully read. Model layers: \n", net.getLayerNames())
#Define the classes
classes = ['human', 'not_human']
while(1):
    # Take each frame
    _, frame = cam.read()
    rows = frame.shape[0]
    cols = frame.shape[1]
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([0, 20, 40])
    upper_blue = np.array([20,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    res = cv.bitwise_and(frame,frame, mask= mask)
    # Bitwise-AND mask and original image
    cv.imshow('mask',mask)
    cv.imshow('frame', frame)
    cv.imshow('res', res) 
    # Blob from image
    blob = cv.dnn.blobFromImage(frame, size=(255, 255), swapRB=True, crop=False)   
    net.setInput(blob)
    cvOut = net.forward()
    print(cvOut.shape)
    # Show the image with a rectagle surrounding the detected objects

    # Cvout is 2d array of 1x1x1x2
    key = cv.waitKey(1)
    #If key pressed is q
    if key == ord('q'):
        #Exit loop
        break
#Release camera
cap.release()
#Destroy all windows


cv.destroyAllWindows()