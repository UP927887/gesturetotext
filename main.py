# Import libraries
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
from tensorflow.keras.models import load_model

# Initialise mediapipe
intialHands = mp.solutions.hands
hands = intialHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
drawingUtil = mp.solutions.drawing_utils

# load models and required files
handModel = load_model('mp_hand_gesture')
gestureFile = open('gesture.names', 'r')
gestureNames = gestureFile.read().split('\n')
gestureFile.close()
print(gestureNames)

# Initialize the webcam and time
cap = cv2.VideoCapture(0)
pTime = 0

while True:
    # Capture frame from webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # FLip webcam
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Initialise gestureName and process gesture results
    result = hands.process(framergb)
    gestureName = ''

    # Take note of each hand position and draw nodes on the hands
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])
            # Drawing landmarks on frames
            drawingUtil.draw_landmarks(frame, handslms, intialHands.HAND_CONNECTIONS)

            # Predict hand gesture
            prediction = handModel.predict([landmarks])
            gestureID = np.argmax(prediction)
            gestureName = gestureNames[gestureID]

    # Show the prediction on the frame
    cv2.putText(frame, gestureName, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)

    # Calculate FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the webcam frame
    cv2.imshow("Gesture Recognition", frame)

    # Break out of the while loop (exit out of the app)
    if cv2.waitKey(1) == ord('q'):
        break

# Stop program
cap.release()
cv2.destroyAllWindows()