import cv2
import numpy as np
from scipy.spatial import distance as dist
import dlib
import time
import os
from scipy.spatial.distance import euclidean
import keyboard

# Load the Haar Cascade face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the Haar Cascade eye detector
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load the Haar Cascade smile detector (for mouth)
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Load the pre-trained model
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define constants
EYE_AR_THRESH = 0.4
EYE_AR_CONSEC_FRAMES = 48
MOUTH_OPEN_THRESH = 0.35
YAWN_THRESHOLD_FRAMES = 25 # Number of frames for a yawn
DEEP_YAWN_THRESHOLD_FRAMES = 50 # Number of frames for a deep yawn

# Initialize variables
yawn_frames = 0

def calculate_ear(eye):
 p2_minus_p6 = dist.euclidean(eye[1], eye[5])
 p3_minus_p5 = dist.euclidean(eye[2], eye[4])
 p1_minus_p4 = dist.euclidean(eye[0], eye[3])
 ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
 return ear

def calculate_mar(mouth):
 p6_minus_p2 = dist.euclidean(mouth[5], mouth[1])
 p10_minus_p4 = dist.euclidean(mouth[9], mouth[3])
 p1_minus_p7 = dist.euclidean(mouth[0], mouth[6])
 mar = (p6_minus_p2 + p10_minus_p4) / (2.0 * p1_minus_p7)
 return mar

# Initialize the video capture object and the detector
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

while True:
 ret, frame = cap.read()
 if not ret:
   break

 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 faces = detector(gray)

 # Print the number of faces detected
 print(f"Number of faces detected: {len(faces)}")

 for face in faces:
   landmarks = predictor(gray, face)

   # Get the landmarks for the left and right eyes
   leftEye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
   rightEye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

   # Calculate the EAR for each eye
   left_ear = calculate_ear(leftEye)
   right_ear = calculate_ear(rightEye)

   # Check to see if the eye aspect ratio is below the blink threshold
   if left_ear < EYE_AR_THRESH or right_ear < EYE_AR_THRESH:
       print("Drowsiness Alert!")

   # Get the landmarks for the mouth
   mouth = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]

   # Calculate the MAR for the mouth
   mar = calculate_mar(mouth)

   # Check to see if the mouth aspect ratio is above the threshold
   if mar > MOUTH_OPEN_THRESH:
       yawn_frames += 1
       if yawn_frames >= DEEP_YAWN_THRESHOLD_FRAMES:
           print("Deep Yawn Alert!")
       elif yawn_frames >= YAWN_THRESHOLD_FRAMES:
           print("Yawn Alert!")
   else:
       yawn_frames = 0

 # Use the Haar Cascade detectors
 faces = face_cascade.detectMultiScale(gray, 1.3, 5)
 eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)
 mouths = smile_cascade.detectMultiScale(gray, 1.7, 11)

 for (x, y, w, h) in faces:
   cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
 for (x, y, w, h) in eyes:
   cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
 for (x, y, w, h) in mouths:
   cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

 # Show the frame
 cv2.imshow('Frame', frame)

 # Break the loop on 'q' key press
 if cv2.waitKey(1) == ord('q'):
   break
 if keyboard.is_pressed('q'):
  break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
