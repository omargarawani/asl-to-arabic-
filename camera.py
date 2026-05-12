# ============================================================
# PATHS — edit these to match your setup
# ============================================================
MODEL_PATH  = "models/inception.h5"
# ============================================================
 
# ============================================================
# SETTINGS
# ============================================================
IMG_SIZE        = 224
HOLD_TIME       = 2.0   # seconds to hold a sign before it's added
CONF_THRESHOLD  = 0.7   # ignore predictions below this confidence
COOLDOWN        = 1.0   # seconds before the same letter can be added again
CAMERA_INDEX    = 0     # 0 = default webcam
# ============================================================
 
import cv2
import numpy as np
import mediapipe as mp
import time
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import cv2
from tensorflow.keras.models import load_model
import numpy as np
 
# ---------- load model ----------
model = load_model(MODEL_PATH)
 
# ---------- labels ----------
labels = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,
          'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,
          'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,'Z':25,'del':26,'nothing':27,'space':28}
labels = {v: k for k, v in labels.items()}
 
# ---------- mediapipe ----------
mp_hands    = mp.solutions.hands
hands       = mp_hands.Hands(max_num_hands=1)
cap         = cv2.VideoCapture(CAMERA_INDEX)
 
# ---------- state ----------
sentence        = ""
current_label   = None
hold_start      = None
last_added_time = 0
 
print("ASL Detection running — press Q to quit, C to clear, B for backspace")
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
 
    detected_label = None
 
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        x = [lm.x for lm in hand.landmark]
        y = [lm.y for lm in hand.landmark]
        x_min = max(0, int(min(x) * w) - 40)
        y_min = max(0, int(min(y) * h) - 40)
        x_max = min(w, int(max(x) * w) + 40)
        y_max = min(h, int(max(y) * h) + 40)
 
        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size > 0:
            h2, w2  = roi.shape[:2]
            size    = max(h2, w2)
            square  = np.zeros((size, size, 3), dtype=np.uint8)
            yo, xo  = (size - h2) // 2, (size - w2) // 2
            square[yo:yo+h2, xo:xo+w2] = roi
 
            img = cv2.resize(square, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)
 
            pred     = model.predict(img, verbose=0)
            class_id = np.argmax(pred)
            conf     = np.max(pred)
 
            if conf >= CONF_THRESHOLD:
                detected_label = labels[class_id]
 
            cv2.putText(frame, f"{labels[class_id]} {conf:.2f}",
                        (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
 
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
 
    # hold-to-add logic
    now = time.time()
    if detected_label is not None:
        if detected_label == current_label:
            if hold_start is not None and (now - hold_start) >= HOLD_TIME:
                if (now - last_added_time) >= COOLDOWN:
                    dl = detected_label.lower()
                    if   dl == "space":           sentence += " "
                    elif dl in ("del", "delete"): sentence  = sentence[:-1]
                    elif dl != "nothing":         sentence += detected_label
                    last_added_time = now
                    hold_start      = now
        else:
            current_label = detected_label
            hold_start    = now
    else:
        current_label = None
        hold_start    = None
 
    # progress bar overlay
    if current_label is not None and hold_start is not None:
        progress = min((now - hold_start) / HOLD_TIME, 1.0)
        bar_w = int(300 * progress)
        cv2.rectangle(frame, (10, 60), (310, 80), (50,50,50), 2)
        cv2.rectangle(frame, (10, 60), (10+bar_w, 80), (0,255,0), -1)
        cv2.putText(frame, f"Holding: {current_label}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
 
    # sentence bar at bottom
    cv2.rectangle(frame, (0, frame.shape[0]-50),
                  (frame.shape[1], frame.shape[0]), (0,0,0), -1)
    cv2.putText(frame, f"Text: {sentence}",
                (10, frame.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
 
    cv2.imshow("ASL Detection", frame)
 
    key = cv2.waitKey(1) & 0xFF
    if   key == ord("q"): break
    elif key == ord("c"): sentence = ""
    elif key == ord("b"): sentence = sentence[:-1]
 
cap.release()
cv2.destroyAllWindows()
with open("sentence.txt", "w", encoding="utf-8") as f:
    f.write(sentence.lower())

print("Final sentence:", sentence)
