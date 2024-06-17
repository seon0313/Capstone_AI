import cv2
import mediapipe as mp
import copy
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
capture = cv2.VideoCapture(0)

size = (640, 480)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

def getFace(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print((int(size[0]/2), int(size[1]/2)), (int(x+(w/2)),int(y+(h/2))))
        cv2.line(frame, (int(size[0]/2), int(size[1]/2)), (int(x+(w/2)),int(y+(h/2))), (255,0,0), 3)
        cv2.imshow('Face Detection', frame)
    except: pass

def getHand(frame):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(hand_landmarks.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
                v = v2 - v1
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
                angle = np.degrees(angle)

                h1 = 1 if np.sum(angle[0:4]) >= 100 else 0
                h2 = 1 if np.sum(angle[3:7]) >= 120 else 0
                h3 = 1 if np.sum(angle[6:10]) >= 120 else 0
                h4 = 1 if np.sum(angle[9:13]) >= 120 else 0
                h5 = 1 if np.sum(angle[12:16]) >= 120 else 0
                print(h1+h2+h3+h4+h5)

                # Draw
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        cv2.imshow('MediaPipe Hands', cv2.flip(frame, 1))

while cv2.waitKey(33) < 0:
    ret, frame = capture.read()
    #cv2.imshow("Camera", frame)
    getFace(copy.deepcopy(frame))
    getHand(copy.deepcopy(frame))

capture.release()
cv2.destroyAllWindows()