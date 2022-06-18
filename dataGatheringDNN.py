import cv2
import numpy as np
from imutils.video import VideoStream
import imutils
import time
from droneConfiguration import *

# load our serialized model from disk
print("loading model...")
net = cv2.dnn.readNetFromCaffe('deployproto.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

face_id = input('\n enter user id (MUST be an integer) -->  ')
numberOfSamples = input('\n enter the number of sample you want to take -->  ')
print("\n Initializing face capture. Look the camera and wait ...")

count = 0

dimensions = [1280, 1024]
# initialize the video stream and give it some time to be ready
print(" starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
#net = cv2.dnn.readNetFromCaffe('deployproto.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

# loop over the frames from the video stream
while True:
    # get the frame
    frame = vs.read()
    frame = imutils.resize(frame, width=dimensions[0], height=dimensions[1])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    startX, startY, endX, endY = findAnyFace(frame, net)
    if startX != 0 and endX != 0:
    # draw the bounding box of the face along with the associated probability
        count += 1
        cv2.imwrite("./images/Users." + str(face_id) + '.' + str(count) + ".jpg", gray[startY:endY, startX:endX])
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(100) & 0xFF

    if key < 100:
        break

    elif count >= numberOfSamples:
         break

cv2.destroyAllWindows()
vs.stop()