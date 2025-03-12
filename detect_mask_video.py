# USAGE
# python detect_mask_video.py

# Import packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from sendmail import *

# Global variables for alert management
ALERT_SENT = False
NO_MASK_FRAMES = 0
REQUIRED_CONSECUTIVE_FRAMES = 5  # 5 consecutive no-mask frames
COOLDOWN_SECONDS = 300            # 5 minutes between alerts
LAST_ALERT_TIME = 0

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))  # Correct mean subtraction
    
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure valid face coordinates
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w-1, endX), min(h-1, endY))
            
            # Skip invalid face regions
            if endX <= startX or endY <= startY:
                continue

            face = frame[startY:endY, startX:endX]
            if face.size == 0:  # Check for empty face ROI
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        # Only predict if valid faces exist
        if faces.shape[0] > 0:
            preds = maskNet.predict(faces, batch_size=32, verbose=0)
        else:
            preds = []
    else:
        preds = []

    return (locs, preds)

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str, default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Load face detector model
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], 
                               "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load mask detector model
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# Initialize video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)  # Camera warm-up




# Add these state variables before the main loop


# ... (keep all existing code until the main while loop)

# loop over the frames from the video stream

ALERT_SENT = False
NO_MASK_FRAMES = 0
last_alert_time = 0

# Then in your loop:
def main():
    global ALERT_SENT, NO_MASK_FRAMES, last_alert_time

while True:
   
    
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    current_time = time.time()

    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    
    # Initialize frame-wide alert status
    any_no_mask = False

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        if mask > withoutMask:
            label = "Mask"
            color = (0, 255, 0)
        else:
            label = "No Mask"
            color = (0, 0, 255)
            any_no_mask = True  # At least one person without mask

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Email alert logic
    if any_no_mask:
        NO_MASK_FRAMES += 1
        # Send email only if:
        # - Haven't sent alert yet
        # - Met consecutive frame threshold
        # - Cooldown period has passed
        if (not ALERT_SENT and 
            NO_MASK_FRAMES >= REQUIRED_CONSECUTIVE_FRAMES and
            (current_time - last_alert_time) > COOLDOWN_SECONDS):
            
            sendmail("Person without mask detected continuously for {} frames".format(
                REQUIRED_CONSECUTIVE_FRAMES))
            ALERT_SENT = True
            last_alert_time = current_time
    else:
        # Reset counters when everyone has masks
        NO_MASK_FRAMES = 0
        ALERT_SENT = False  # Reset for new detections

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# ... (keep cleanup code)

# do a bit of cleanup
try:
    # Your existing video processing loop
    while True:
        # Your code for video processing and detecting masks
        # If a person doesn't wear a mask, the sendmail function will be called
        sendmail("This person has no mask")
except KeyboardInterrupt:
    print("Program interrupted. Exiting...")
finally:
    cv2.destroyAllWindows()
    vs.stop()
