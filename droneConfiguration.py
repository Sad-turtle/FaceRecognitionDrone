from djitellopy import Tello
import cv2
import numpy as np

def initializeTello():
    puppet = Tello()
    puppet.connect()
    puppet.for_back_velocity = 0
    puppet.left_right_velocity = 0
    puppet.up_down_velocity = 0
    puppet.yaw_velocity = 0
    puppet.speed = 0
    print(puppet.get_battery())
    puppet.streamoff()
    puppet.streamon()
    return puppet

def telloGetFrame(puppet, dimensions):
    myFrame = puppet.get_frame_read()
    myFrame = myFrame.frame
    img = cv2.resize(myFrame, (dimensions[0], dimensions[1]))
    return img

def findAnyFace(img, net):
    # grab the frame dimensions and convert it to a blob
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceDetected = False
    startX, startY, endX, endY = 0, 0, 0, 0
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        faceConfidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if faceConfidence < 0.5:
            faceDetected = True
            continue
        else:
            faceDetected = False

        # compute the (x, y)-coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

    if faceDetected:
        return startX, startY, endX, endY
    else:
        return 0, 0, 0, 0


def trackFace(puppet, vectorDistance, safety_zone, speed):

    if vectorDistance[0] < -safety_zone[0]:
        puppet.yaw_velocity = speed

    elif vectorDistance[0] > safety_zone[0]:
        puppet.yaw_velocity = -speed

    else:
        puppet.yaw_velocity = 0

        # for up & down
    if vectorDistance[1] > safety_zone[1]:
        puppet.up_down_velocity = speed
    elif vectorDistance[1] < -safety_zone[1]:
        puppet.up_down_velocity = -speed
    else:
        puppet.up_down_velocity = 0

        # for forward back
    if vectorDistance[2] > 30:
        puppet.for_back_velocity = speed
    elif vectorDistance[2] < -30:
        puppet.for_back_velocity = -speed
    else:
        puppet.for_back_velocity = 0

    update(puppet)

def update(puppet):
    if puppet.send_rc_control:
        puppet.send_rc_control(puppet.left_right_velocity,
                               puppet.for_back_velocity,
                               puppet.up_down_velocity,
                               puppet.yaw_velocity)

def manualControl(puppet, speed):
    # S & W to fly forward & back
    key = cv2.waitKey(1)
    if key == ord('w'):
        puppet.for_back_velocity = int(speed)
    elif key == ord('s'):
        puppet.for_back_velocity = -int(speed)
    else:
        puppet.for_back_velocity = 0

    # a & d to pan left & right
    if key == ord('d'):
        puppet.yaw_velocity = int(speed)
    elif key == ord('a'):
        puppet.yaw_velocity = -int(speed)
    else:
        puppet.yaw_velocity = 0

    # Q & E to fly up & down
    if key == ord('e'):
        puppet.up_down_velocity = int(speed)
    elif key == ord('q'):
        puppet.up_down_velocity = -int(speed)
    else:
        puppet.up_down_velocity = 0

    # c & z to fly left & right
    if key == ord('c'):
        puppet.left_right_velocity = int(speed)
    elif key == ord('z'):
        puppet.left_right_velocity = -int(speed)
    else:
        puppet.left_right_velocity = 0
    update(puppet)



