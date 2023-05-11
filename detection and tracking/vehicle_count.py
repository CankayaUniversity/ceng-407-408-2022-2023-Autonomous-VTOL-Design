# TechVidvan Vehicle counting and Classification

# Import necessary packages

import cv2
import csv
import collections
import numpy as np
from tracker import *

# Initialize Tracker
tracker = EuclideanDistTracker()
trackers = cv2.legacy.MultiTracker_create() # initialize a multi-tracker object

# Initialize the videocapture object
cap = cv2.VideoCapture("C:/Users/Monster/Desktop/video.mp4")
input_size = 416

# Detection confidence threshold
confThreshold =0.2
nmsThreshold= 0.2

font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2

# Middle cross line position
middle_line_position = 225   
up_line_position = middle_line_position - 15
down_line_position = middle_line_position + 15


# Store Coco Names in a list
classesFile = "C:/Users/Monster/Desktop/vehicle-detection-classification-opencv/coco.names"
classNames = open(classesFile).read().strip().split('\n')
print(classNames)
print(len(classNames))

# class index for our required detection classes
required_class_index = [2]

detected_classNames = []


## Model Files
modelConfiguration = 'C:/Users/Monster/Desktop/vehicle-detection-classification-opencv/yolov3-320.cfg'
modelWeigheights = 'C:/Users/Monster/Desktop/vehicle-detection-classification-opencv/yolov3-320.weights'

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# Configure the network backend

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')


# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy


# Function for finding the detected objects from the network output
def postProcess(outputs,img):
    global detected_classNames 
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            ih, iw, channels = img.shape
            if classId in required_class_index:
                if confidence > confThreshold:
                    # print(classId)
                    w,h = int(det[2]*width) , int(det[3]*height)
                    x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                    boxes.append([x,y,w,h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))
    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    # print(classIds)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            # print(x,y,w,h)

            color = [int(c) for c in colors[classIds[i]]]
            name = classNames[classIds[i]]
            detected_classNames.append(name)
            # Draw classname and confidence score 
            cv2.putText(img,f'{name.upper()} {int(confidence_scores[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw bounding rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            detection.append([x, y, w, h, required_class_index.index(classIds[i])])
def realTime():
    while True:
        success, frame = cap.read()
        frame = cv2.resize(frame,(0,0),None,0.5,0.5)
        ih, iw, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

        # Set the input of the network
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
        # Feed data to the network
        outputs = net.forward(outputNames)
    
        # Find the objects from the network output
        postProcess(outputs,frame)
        if frame is None:
            break
        frame = cv2.resize(frame,(1090,600))

        success, boxes = trackers.update(frame)

        for i, box in enumerate(boxes):
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'TRACKING {i}',(x+10,y-3),cv2.FONT_HERSHEY_PLAIN,1.5,(255,255,0),2)

        #cv2.imshow('Frame', frame)
        k = cv2.waitKey(30)

        if k == ord('s'):
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            roi = cv2.selectROI('Frame', frame, fromCenter=False,
                                showCrosshair=True)
            tracker = cv2.legacy.TrackerKCF_create() # initialize a new tracker for the selected ROI
            trackers.add(tracker, frame, roi)

        # Show the frames
        cv2.imshow('Output',frame)

        if cv2.waitKey(1) == ord('q'):
            break
        
        
    # Finally realese the capture object and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    realTime()