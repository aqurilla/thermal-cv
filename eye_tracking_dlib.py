# Import the packages necessary for the operation
import argparse
import datetime
import imutils
import time
import cv2
import numpy
import dlib

# Parse arguments using the argument parser
parser = argparse.ArgumentParser(description="Simple Motion Tracker")
parser.add_argument("-v", "--video", help="path to video file given as input")
args = vars(parser.parse_args())

# If no video argument is supplied, we are using camera
if args.get("video", None) is None:
    camera = cv2.VideoCapture(0)
    # Pause program for 0.25 secs
    time.sleep(1)

# If a video argument is supplied, we are using video
else:
    camera = cv2.VideoCapture(args["video"])

# Initialize detector path
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start camera feed until keypress
while True:
# c = 0
# while c != 1:
    (grabbed, frame) = camera.read()

    if not grabbed: break

    # Resize and convert to grayscale
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        
        # Specify the region of interest for the face
        rect = dlib.rectangle(long(x),long(y),long(x+w),long(y+h))
        # Detect facial features
        shape = predictor(frame, rect)

        #mat = numpy.matrix([[p.x,p.y] for p in shape.parts()])

        for p in shape.parts():
            pos = (p.x,p.y)
            cv2.circle(frame, pos, 1, (0, 255, 0), -1)


    cv2.imshow("Img", frame)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the loop
    if key == ord("q"): break

    # Test
    # c = 1

camera.release()
cv2.destroyAllWindows()









