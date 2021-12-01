import yawndetection as ydetect
import cv2
import tensorflow as tf
import os
from pygame import mixer
import numpy as np
import time
from tensorflow.keras.models import Model,load_model
#import tensorflow_hub as hub
import argparse
import posenet
from keras import backend as K
import db_connect
from datetime import datetime

parser = argparse.ArgumentParser() # Argumentum feldolgozása
parser.add_argument('--model', type=int, default=101) # A model definiálása
parser.add_argument('--cam_id', type=int, default=0) # Kamera ID
parser.add_argument('--cam_width', type=int, default=1280) # Kamerakép szélessége
parser.add_argument('--cam_height', type=int, default=720) # Kamerakép magassága
parser.add_argument('--scale_factor', type=float, default=0.7125) # Skálázási faktor
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera") # Video fájl megadása
args = parser.parse_args()

def main():
    face = cv2.CascadeClassifier('./input/cascade/haarcascade_frontalface_default.xml') # Arc kaszkádoló fájl
    leye = cv2.CascadeClassifier('./input/cascade/haarcascade_lefteye_2splits.xml') # Bal szem kaszkádoló fájl
    reye = cv2.CascadeClassifier('./input/cascade/haarcascade_righteye_2splits.xml') # Jobb szem kaszkádoló fájl
    mixer.init() # Hang lejátszó betöltése
    sound = mixer.Sound('./sound/emergency.wav') # Hangfájl helyének megadása
    label = ['Close', 'Open'] # Címkék megadása (Csukott, Nyitott)
    yawns = 0 # Ásítás számolása
    yawn_status = False # Ásítás állapota
    reye_pred = [99] # Jobb szem detektáláshoz tömb létrehozása
    leye_pred = [99] # Bal szem detektáláshoz tömb létrehozása
    count = 0 # Számoláshoz segédváltozó
    score = 0 # Szemek állapotához segédváltozó
    thicc = 2 # Keret vastagságához segédváltozó
    alerted = False
    with tf.Graph().as_default(): # Gráf beállítása alapértelmezettnek
        with tf.compat.v1.Session() as sess: # "Session" létrehozása
            eye_model = load_model('./keras/model/eye_model.h5') # Keras modell betöltése
            model_cfg, model_outputs = posenet.load_model(args.model, sess) # PoseNet Modell betöltése
            output_stride = model_cfg['output_stride'] # Kimeneti lépésszám

            if args.file is not None: # Ha argumentumként videófájl van megadva
                cap = cv2.VideoCapture(args.file)

            else:
                cap = cv2.VideoCapture(args.cam_id) # Ha nincs, akkor alapértelmezett webkamera
            #print("Cam_width: ", args.cam_width)
            #cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width) # Képkocka szélességének beállítása argumentum alapján
            #print("Cam_height: ", args.cam_height)
            #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height) # Képkocka magasságának beállítása argumentum alapján
            start = time.time() # Kezdési idő tárolása
            ref_sec = start # Viszonyítási másodperc
            frame_count = 0 # Képkocka számolását segítő változó
            while True: # Amíg igaz
                sec_now = time.time() # Aktuális idő tárolása
                elapsed_sec = sec_now - ref_sec # Eltelt másodperc tárolása
                if (elapsed_sec > 30 and alerted == False): # Ha az eltelt másodperc, több mint 30, és nem történt riasztás
                    sql = "DELETE FROM szem WHERE Idopont > DATE_ADD(NOW(), INTERVAL -30 SECOND);" # SQL kód definiálása
                    db_connect.delete_data(sql) # SQL-ből törlés függvénnyel
                    sql = "DELETE FROM szaj WHERE Idopont > DATE_ADD(NOW(), INTERVAL -30 SECOND);" # SQL kód definiálása
                    db_connect.delete_data(sql) # SQL-ből törlés függvénnyel
                    sql = "DELETE FROM felsotest WHERE Idopont > DATE_ADD(NOW(), INTERVAL -30 SECOND);" # SQL kód definiálása
                    db_connect.delete_data(sql) # SQL-ből törlés függvénnyel
                    ref_sec = time.time() # Viszonyítási másodpercnek a mostani idő átadása
                elif (elapsed_sec > 30 and alerted == True): # Ha az eltelt másodperc, több mint 30, és történt riasztás
                    alerted = False # Riasztási állapot hamisra állítása
                    ref_sec = time.time() # Viszonyítási másodpercnek a mostani idő átadása
                    
                ret, frame = cap.read() # Visszatérési értéket a "return",valamint a képkockát elmenti a "frame" változóba
                frame = cv2.resize(frame, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA) # A képkocka felbontásának csökkentése
                height, width = frame.shape[:2] # Képkocka méretének értékei változóba mentése
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Képkocka szürke árnyalatossá konvertálása
                faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25)) # Arc detektálása
                left_eye = leye.detectMultiScale(gray) # Bal szem detektálása
                right_eye = reye.detectMultiScale(gray) # Jobb szem detektálása
                cv2.rectangle(frame, (0, height - 50), (500, height), (0, 0, 0), thickness=cv2.FILLED) # Alsó fekete téglalap
                for (x, y, w, h) in faces: # A detektált arc köré négyszög rajzolása
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

                image_landmarks, lip_distance = ydetect.mouth_open(frame) # Ásítás detektálása
                prev_yawn_status = yawn_status # Előző ásítási állapot elmentése

                if (lip_distance > 20): # Ha ajkak közti távolság nagyobb mint 20
                    yawn_status = True # Ásítási állapot igaz
                    sql = "INSERT INTO szaj (SzajAllapot, Idopont) VALUES (%s , %s )" # SQL kód definiálása
                    values = ('Asitas', datetime.now()) # Értékek meghatározása
                    db_connect.insert_into_table(sql, values) # SQL-be mentés függvénnyel
                    cv2.putText(frame, # Szövegdoboz elhelyezése a képen
                                "The driver is yawning", # Szöveg
                                (200, 20), # Koordináta
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, # Betűtípus
                                0.7, # Betűméret
                                (0, 0, 255), # Betűszín
                                1) # Vastagság

                    output_text = "Number of Yawns: " + str(yawns + 1) # Kimeneti szöveg
                    #Szöveg elhelyezés
                    cv2.putText(frame, output_text, (180, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7,
                                (255, 255, 255), 1)
                else:
                    yawn_status = False # Ha kisebb mint 25, ásítás állapota hamis
                    sql = "INSERT INTO szaj (SzajAllapot, Idopont) VALUES (%s , %s )" # SQL kód definiálása
                    values = ('Csukott', datetime.now()) # Értékek meghatározása
                    db_connect.insert_into_table(sql, values) # SQL-be mentés függvénnyel

                if (prev_yawn_status == True and yawn_status == False): # Ha az előző ásítási állapot igaz volt, és a mostani állapot hamis
                    yawns += 1 # Ásítás számának növelése

                for (x, y, w, h) in right_eye: # For ciklus a detektált jobb szem képen
                    r_eye = frame[y:y + h, x:x + w] # Bemeneti kép koordinátáinak módosítása szélesség, és hossz hozzáadásával
                    count = count + 1 # Segédváltozó növelése
                    r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY) # Kép átkonvertálása szürkeárnyalatossá
                    r_eye = cv2.resize(r_eye, (24, 24)) # Kép átméretezése 24x24-es méretté
                    r_eye = r_eye / 255 # Kép értékeit elosztjuk 255-el
                    r_eye = r_eye.reshape(24, 24, -1) # Átformálás
                    r_eye = np.expand_dims(r_eye, axis=0) # Dimenziók növelése
                    reye_pred = np.argmax(eye_model.predict(r_eye), axis=1) # Maximális érték kiszámítása megadott tengely mentén, modell alkalmazása
                    if (reye_pred[0] == 1): # Ha detektálás 1
                        lbl = 'Open' # Nyitott
                    if (reye_pred[0] == 0): # Ha detektálás 0
                        lbl = 'Closed' # Csukott
                    break
                # Bal szem detektálása
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

                if (reye_pred[0] == 0 and leye_pred[0] == 0): # Ha a jobb, és bal szem csukva
                    score = score + 1 # Segédváltozó növelése
                    sql = "INSERT INTO szem (SzemAllapot, Idopont) VALUES (%s , %s )" # SQL kód definiálása
                    values = ('Csukott', datetime.now()) # Értékek meghatározása
                    db_connect.insert_into_table(sql, values) # SQL-be mentés függvénnyel
                    #Szövegdoboz elhelyezése
                    cv2.putText(frame, "Closed", (10, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 255, 255),
                                1, cv2.LINE_AA)
                else: # Ha mindkettő nyitva
                    score = score - 2 # Segédváltozó csökkentése kettővel
                    sql = "INSERT INTO szem (SzemAllapot, Idopont) VALUES (%s , %s )" # SQL kód definiálása
                    values = ('Nyitott', datetime.now()) # Értékek meghatározása
                    db_connect.insert_into_table(sql, values) # SQL-be mentés függvénnyel
                    #Szövegdoboz elhelyezése
                    cv2.putText(frame, "Opened", (10, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 255, 255),
                                1, cv2.LINE_AA)
                if (score < 0): # Ha kisebb mint 0
                    score = 0 # 0-val lesz egyelő
                # Szövegdoboz elhelyezése
                cv2.putText(frame, 'Score:' + str(score), (80, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7,
                            (255, 255, 255), 1, cv2.LINE_AA)
                if (score > 10): #  Ha a "score" nagyobb, mint 10
                    try:
                        sound.play() # Hangjelzés lejátszása

                    except:
                        pass # Ha nem sikerül, akkor nem játssza le
                    if (thicc < 16): # Ha a keret vastagsága kisebb, mint 16
                        thicc = thicc + 2 # Vastagság növelése kettővel
                    else: # Ha nem kisebb
                        thicc = thicc - 2 # Vastagság csökkentése kettővel
                        if (thicc < 2):  # Ha kisebb mint kettő
                            thicc = 2 # Vastagság értéke kettő
                    # Négyszög rajzolása
                    cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
                # Bemeneti kép beolvasása PoseNet
                input_image, display_image, output_scale = posenet.read_cap(
                    frame, # Bemeneti kép
                    scale_factor=args.scale_factor, # Átméretezési arány
                    output_stride=output_stride) # Kimeneti lépésszám

                # Munkamenet (Session) futtatása
                heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                    model_outputs, # Modell kimenet
                    feed_dict={'image:0': input_image} # Gráf elemeit értékekre képzi le
                )

                # Felsőtest pózdetektálás
                pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                    heatmaps_result.squeeze(axis=0), # Hőtérkép módosítása
                    offsets_result.squeeze(axis=0), # Ellensúlyok módosítása
                    displacement_fwd_result.squeeze(axis=0), # "Forward convolutions" elmozdulás eredménye
                    displacement_bwd_result.squeeze(axis=0), # "Backward convolutions" elmozdulás eredménye
                    output_stride=output_stride, # Kimeneti lépésszám
                    max_pose_detections=10, # Maximum póz detektálása
                    min_pose_score=0.15) # Minimum póz érték

                keypoint_coords *= output_scale # Kulcspont koordináták megszorzása a kimeneti skálával

                #Kimeneti kép
                frame = posenet.draw_skel_and_kp( # PoseNet csontváz rajzolása
                    display_image, # Kimeneti kép
                    pose_scores, # Póz értékek
                    keypoint_scores, # Kulcspont értékek
                    keypoint_coords, # Kulcspontok koordinátái
                    min_pose_score=0.15, # Minimum póz érték
                    min_part_score=0.1) # Minimum testrész érték
                #out.write(frame)
                frame_count = frame_count + 1
                cv2.imshow('posenet', frame) # Kimeneti kép megjelenítése
                if cv2.waitKey(1) & 0xFF == ord('q'): # Ha a 'Q' gomb megnyomásra kerül
                    break # Kilép a while ciklusból


if __name__ == "__main__":
    main()