import cv2
import numpy as np
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import streamlit as st
from PIL import Image
import numpy as np
def load_models():
    # Load the face detector model
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # Load the face mask detector model
    print("[INFO] loading face mask detector model...")
    model = load_model("mask_detector.model")
    
    return net, model

def detect_mask_on_image(image, net, model):
    # Get image dimensions
    (h, w) = image.shape[:2]
    
    # Construct a blob from the image and perform face detection
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Process each face detection
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            # Get face bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure the bounding boxes are within frame dimensions
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            # Process face region (ROI)
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # Predict mask vs no mask
            (mask, withoutMask) = model.predict(face)[0]
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    return image

def process_image(image_path):
    # Load models
    net, model = load_models()

    # Read and process image
    image = cv2.imread(image_path)
    image = detect_mask_on_image(image, net, model)
    return image


def mask_detection():
    st.title("Face Mask Detection")

    activities = ["Image", "Webcam"]
    st.set_option('deprecation.showfileUploaderEncoding', False)
    choice = st.sidebar.selectbox("Mask Detection on?", activities)

    if choice == 'Image':
        st.subheader("Detection on Image")
        image_file = st.file_uploader("Upload Image", type=['jpg'])
        if image_file is not None:
            # Save the uploaded image
            uploaded_image = Image.open(image_file)
            image_path = './images/out.jpg'
            uploaded_image.save(image_path)
            st.image(uploaded_image, caption='Image uploaded successfully', use_column_width=True)

            if st.button('Process'):
                # Process the uploaded image and get the result
                result_image = process_image(image_path)
                
                # Display the result
                result_pil = Image.fromarray(result_image)
                st.image(result_pil, caption="Processed Image", use_column_width=True)

    if choice == 'Webcam':
        st.subheader("Detection on Webcam")
        st.text("This feature will be available soon")
