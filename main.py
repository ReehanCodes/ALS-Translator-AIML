#Declaring dependencies 
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp


mp_holistic = mp.solutions.holistic     #holistics model (make detection)
mp_drawing = mp.solutions.drawing_utils #drawing utilities (draw detection)
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #converting color channel bgr 2 rgb (mediapipe only uses rgb but opencv captures bgr)
    image.flags.writeable = False                  #image is not writable
    results = model.process(image)                 #use mediapipe to detect frame from opencv
    image.flags.writeable = True                    #make image writable again
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #converting color channel rgb 2 bgr (so opencv can use)
    return image, results











cap = cv2.VideoCapture(0)

#Access mediapipe model
with mp_holistic.Holistic(min_detection_confidence= 0.5, min_tracking_confidence= 0.5) as holistic:    #initial detection tracking
    while cap.isOpened():   
        ret, frame = cap.read()     #Read feed frames


        #Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        cv2.imshow('Hey', frame)    #Display to screen
        #Break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()           #Release webcam
    cv2.destroyAllWindows() #Break window






