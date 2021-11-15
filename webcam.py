import yawndetection as ydetect
import cv2
import os
from pygame import mixer
import numpy as np
import time
from tensorflow.keras.models import load_model

label = ['Close', 'Open']
eye_model = load_model('./eye_model.h5')

face = cv2.CascadeClassifier('./input/cascade/haarcascade_frontalface_default.xml')
leye = cv2.CascadeClassifier('./input/cascade/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('./input/cascade/haarcascade_righteye_2splits.xml')

mixer.init()
sound = mixer.Sound('./emergency.wav')

videocap = cv2.VideoCapture(0)
yawns = 0
yawn_status = False
reye_pred = [99]
leye_pred = [99]
count = 0
ycount = 0
score = 0
thicc = 2

while (True):
    ret, frame = videocap.read()

    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)
    cv2.rectangle(frame, (0, height - 50), (500, height), (0, 0, 0), thickness=cv2.FILLED)

    image_landmarks, lip_distance = ydetect.mouth_open(frame)
    prev_yawn_status = yawn_status

    if (lip_distance > 25):
        yawn_status = True
        cv2.putText(frame, "The driver is yawning", (350, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        output_text = "Number of Yawns: " + str(yawns + 1)
        cv2.putText(frame, output_text, (200, height-20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
    else:
        yawn_status = False

    if (prev_yawn_status == True and yawn_status == False):
        yawns += 1

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count = count + 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        reye_pred = np.argmax(eye_model.predict(r_eye),axis=1)
        if (reye_pred[0] == 1):
            lbl = 'Open'
        if (reye_pred[0] == 0):
            lbl = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count = count + 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        leye_pred = np.argmax(eye_model.predict(l_eye),axis=1)
        if (leye_pred[0] == 1):
            lbl = 'Open'
        if (leye_pred[0] == 0):
            lbl = 'Closed'
        break

    if (reye_pred[0] == 0 and leye_pred[0] == 0):
        score = score + 1
        cv2.putText(frame, "Closed", (10, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score = score - 1
        cv2.putText(frame, "Opened", (10, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if (score < 0):
        score = 0
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if (score > 15):
        try:
            sound.play()

        except:
            pass
        if (thicc < 16):
            thicc = thicc + 2
        else:
            thicc = thicc - 2
            if (thicc < 2):
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
videocap.release()
cv2.destroyAllWindows()