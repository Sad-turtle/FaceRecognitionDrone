import numpy as np
import time
import cv2
from droneConfiguration import *

# parameters for drone
safetyZone = [90, 90]
speed = 20
puppet = initializeTello()
targetFaceSize = 304
startCounter = 1
# These are our center dimensions
dimensions = [640, 480]
cWidth = dimensions[0]//2
cHeight = dimensions[1]//2

net = cv2.dnn.readNetFromCaffe('deployproto.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainedModel.yml')

names = ['Unknown', 'Yernar', 'Mate']
font = cv2.FONT_HERSHEY_SIMPLEX

faceId = 0

# Flight
if startCounter == 0:
    puppet.takeoff()
    startCounter = 1

# loop over the frames from the video stream
while True:
    # step 1 get image from camera
    frame = telloGetFrame(puppet, dimensions)
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # get coordinates of the face
    startX, startY, endX, endY = findAnyFace(frame, net)

    # these are our target coordinates
    target_x = int((endX + startX) / 2)
    target_y = int((endY + startY) / 2)
    end_size = (endX - startX) * 2

    # We find distance vector
    if target_x != 0 or target_y != 0 or end_size != 0:
        vectorDistance = np.array((cWidth, cHeight, targetFaceSize)) - np.array((target_x, target_y, end_size))
    else:
        vectorDistance = [0, 0, 0]
    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    faceId, matchConfidence = recognizer.predict(imgGray)
    if matchConfidence <= 80:
        faceName = names[faceId]
        matchConfidence = "  {0}%".format(round(100 - matchConfidence))
        trackFace(puppet, vectorDistance, safetyZone, speed)
    else:
        # Unknown Face
        faceName = names[0]
        matchConfidence = "  {0}%".format(round(matchConfidence))

    cv2.circle(frame, (cWidth, cHeight), 10, (0, 0, 255), 2)
    cv2.putText(frame, str(faceName), (startX + 5, startY - 5), font, 1, (255, 255, 255), 2)
    cv2.putText(frame, str(matchConfidence), (startX + 5, endY - 5), font, 1, (255, 255, 0), 1)

    # Draw the estimated drone vector position in relation to face bounding box
    cv2.putText(frame, str(vectorDistance), (0, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # land if ESC is pressed
        puppet.land()
        break


