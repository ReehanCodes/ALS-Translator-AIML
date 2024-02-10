import cv2
import numpy as np




#capturing webcam & using haar cascade object detection
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# retrieving default camera settings of default webcam
default_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
default_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
default_fps = cap.get(cv2.CAP_PROP_FPS)



cap.set(cv2.CAP_PROP_FRAME_WIDTH, default_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, default_height)
cap.set(cv2.CAP_PROP_FPS, default_fps)




while True:
    # capture every frame and make it gray
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect faces in gray frames
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(90, 90))
    
    # draw rectangles in detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 5, 150), 10)





    # Display the captured frame
    cv2.imshow('Webcam', frame)
    
    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()