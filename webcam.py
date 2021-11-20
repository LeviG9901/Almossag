import yawndetection as ydetect
import cv2
import os
from pygame import mixer
import numpy as np
import time
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import tensorflow as tf
import argparse
import posenet
from keras import backend as K

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

def main():
    face = cv2.CascadeClassifier('./input/cascade/haarcascade_frontalface_default.xml')
    leye = cv2.CascadeClassifier('./input/cascade/haarcascade_lefteye_2splits.xml')
    reye = cv2.CascadeClassifier('./input/cascade/haarcascade_righteye_2splits.xml')
    mixer.init()
    sound = mixer.Sound('./sound/emergency.wav')
    label = ['Close', 'Open']
    yawns = 0
    yawn_status = False
    reye_pred = [99]
    leye_pred = [99]
    count = 0
    ycount = 0
    score = 0
    thicc = 2
    var = 0
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            eye_model = load_model('./keras/model/eye_model.h5')
            model_cfg, model_outputs = posenet.load_model(args.model, sess)
            output_stride = model_cfg['output_stride']

            if args.file is not None:
                cap = cv2.VideoCapture(args.file)
            else:
                cap = cv2.VideoCapture(args.cam_id)

            cap.set(3, args.cam_width)
            cap.set(4, args.cam_height)

            start = time.time()
            frame_count = 0
            while True:

                ret, frame = cap.read()
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
                    cv2.putText(frame, "The driver is yawning", (350, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (0, 0, 255), 1)
                    output_text = "Number of Yawns: " + str(yawns + 1)
                    cv2.putText(frame, output_text, (200, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (255, 255, 255), 1)
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
                    reye_pred = np.argmax(eye_model.predict(r_eye), axis=1)
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
                    leye_pred = np.argmax(eye_model.predict(l_eye), axis=1)
                    if (leye_pred[0] == 1):
                        lbl = 'Open'
                    if (leye_pred[0] == 0):
                        lbl = 'Closed'
                    break

                if (reye_pred[0] == 0 and leye_pred[0] == 0):
                    score = score + 1
                    cv2.putText(frame, "Closed", (10, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255),
                                1, cv2.LINE_AA)
                else:
                    score = score - 2
                    cv2.putText(frame, "Opened", (10, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255),
                                1, cv2.LINE_AA)

                if (score < 0):
                    score = 0
                cv2.putText(frame, 'Score:' + str(score), (100, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (255, 255, 255), 1, cv2.LINE_AA)
                if (score > 10):
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

                input_image, display_image, output_scale = posenet.read_cap(
                    frame, scale_factor=args.scale_factor, output_stride=output_stride)

                heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                    model_outputs,
                    feed_dict={'image:0': input_image}
                )

                pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                    heatmaps_result.squeeze(axis=0),
                    offsets_result.squeeze(axis=0),
                    displacement_fwd_result.squeeze(axis=0),
                    displacement_bwd_result.squeeze(axis=0),
                    output_stride=output_stride,
                    max_pose_detections=10,
                    min_pose_score=0.15)

                keypoint_coords *= output_scale

                frame = posenet.draw_skel_and_kp(
                    display_image, pose_scores, keypoint_scores, keypoint_coords,
                    min_pose_score=0.15, min_part_score=0.1)

                cv2.imshow('posenet', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


if __name__ == "__main__":
    main()