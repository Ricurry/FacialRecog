import cv2 as cv
import numpy as np
import pyvirtualcam
import pyautogui
import ctypes
from ctypes import wintypes 
# Set failsafe to false
pyautogui.FAILSAFE = False
# Set the refresh of pyautogui to 0.01 seconds
pyautogui.PAUSE = 0
cam = cv.VideoCapture(0)
#Set camera resolution to 256x256
x = user32.GetSystemMetrics(0)
y = user32.GetSystemMetrics(1)

cam.set(3, 255)
cam.set(4, 255)
net = cv.dnn.readNetFromTensorflow('frozen_graph.pb')
print("OpenCV model was successfully read. Model layers: \n", net.getLayerNames())
classes = ['human', 'not_human']
while(1):
    # Take each frame
    _, frame = cam.read()
    # Frame should be rotated 90 degrees
    frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
    rows = frame.shape[0]
    cols = frame.shape[1]
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # Upper and lower limits of the color of the hand
    lower = np.array([0, 20, 40])
    upper = np.array([20,255,255]) # color key
    mask = cv.inRange(hsv, lower, upper) 
    res = cv.bitwise_and(frame,frame, mask= mask) # using mask over the frame.
    cv.imshow('mask',mask)
    cv.imshow('frame', frame)
    blob = cv.dnn.blobFromImage(frame, size=(255, 255), swapRB=True, crop=False)  # Using the deep neural network to make a initial prediction.
    net.setInput(blob) # Read from the tensorflow generated model the image.
    cvOut = net.forward() # Go to next layer
    print("Shape of cvOut:", cvOut.shape) 
    print("Contents of cvOut:")
    print(cvOut)


    detection = cvOut
    score = cvOut
    threshold = 0.05

    class_name = 'human' if score > threshold else 'not_human' # Checks the mean score and if its higher than the threshold it will use the the average positions of the x and y to move the mouse because we only need to make one prediction in order to lessen the use of AI. :3
    if np.mean(mask) == 10:
        pyautogui.click(duration=0.25)

    if score > 0.05:
        class_index = int(detection)
        print(detection.shape)
        bounding_size = detection * 3
        center_x = int(bounding_size * cols * 1.5)
        # bound sizes
        center_y = int(bounding_size * rows * 1.5)
        width = int(bounding_size * cols)
        height = int(bounding_size * rows)
        topleft_x = int(center_x - width / 2)
        topleft_y = int(center_y - height / 2)
        bottomright_x = topleft_x + width
        bottomright_y = topleft_y + height

        cv.rectangle(frame, (topleft_x, topleft_y), (bottomright_x, bottomright_y), (0, 255, 0), thickness=2)
        cv.putText(frame, class_name, (topleft_x, topleft_y + 20), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        print("X position:", center_x)
        print("Y position:", center_y)
        y_coords, x_coords = np.where(mask == 255)

        average_x = np.mean(x_coords)
        average_y = np.mean(y_coords)

        # Move mouse to average x and y
        # divides to position of monitor. We can use user32.getSystemMetrics(0) and user32.getSystemMetrics(1).
        pyautogui.moveTo(average_x * x/500, average_y * y/500)
 
    cv.imshow('frame', frame)

    # Cvout is 2d array of 1x1x1x2
    key = cv.waitKey(1)
    #If key pressed is q
    if key == ord('q'):
        #Exit loop
        break
#Release camera
cam.release()
#Destroy all windows

# We are finished.
# I finally completed it.
cv.destroyAllWindows()