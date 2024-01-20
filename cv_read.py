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
    # Blob from image
    blob = cv.dnn.blobFromImage(frame, size=(255, 255), swapRB=True, crop=False)   
    net.setInput(blob)
    cvOut = net.forward()
    print(cvOut.shape   )
    # Make box around 4 dimensional object
    # Blob from image
    blob = cv.dnn.blobFromImage(frame, size=(255, 255), swapRB=True, crop=False)   
    net.setInput(blob)
    cvOut = net.forward()

    # Print information about cvOut
    print("Shape of cvOut:", cvOut.shape)
    print("Contents of cvOut:")
    print(cvOut)

# Extract information from the single detection
    detection = cvOut[0, 0, 0, 0]
    score = cvOut
    # Assuming threshold for considering as 'human'
    threshold = 0.05

    # Assign class name based on the threshold
    class_name = 'human' if score > threshold else 'not_human'
# Process the detection if the score is above a threshold
    print(score)
    if score > 0.05:
        # Directly use scalar values without indexing
        class_index = int(detection)
        bounding_size = detection * 5
        # Center the bounding box around the center of the image
        # The 255 is the size of the image

        center_x = int(bounding_size* cols)
        center_y = int(bounding_size * rows)
        width = int(bounding_size * cols)
        height = int(bounding_size * rows)
        topleft_x = int(center_x - width / 2)
        topleft_y = int(center_y - height / 2)
        # Draw rectangle
        cv.rectangle(frame, (topleft_x, topleft_y), (topleft_x + width, topleft_y + height), (0, 255, 0), thickness=2)
        # Put text on image
        cv.putText(frame, class_name, (topleft_x, topleft_y + 20), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        print("X position:", center_x)
        print("Y position:", center_y)

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