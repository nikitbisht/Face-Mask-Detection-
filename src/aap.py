import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model


mask_detect = {
    0:'Mask Detected',
    1:'No Mask Deteceted'
}

haar = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
def detect_face(img):
    coords = haar.detectMultiScale(img)
    return coords

def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224)) 
    image_array = np.array(image)  
    image_array = image_array / 255.0 
    test_input = np.expand_dims(image_array, axis=0) 
    return test_input

def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)  
    return preprocess_image(image)  

model = load_model('model/face_mask_detect.h5')
st.title("☺ Face Mask Detection System")

# Sidebar for mode selection
mode = st.sidebar.selectbox("Choose Mode", ("Upload Image", "Real-Time Camera"))

if mode == "Upload Image":
    uploaded_image = st.file_uploader("Upload your image here:", type=["jpg", "jpeg", "png", "jfif"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=False)

        if st.button("Predict"):
            test_input = preprocess_image(image)
            prediction_array = model.predict(test_input)
            y_pred = ((prediction_array >= 0.5).astype(int)).item()
            
            st.markdown(
                f"<h2 style='text-align: center; color: #FFFF00;'>"
                f"Emotion detected: <span style='font-weight: bold; font-size: 48px;'>{mask_detect[y_pred]}</span> </h2>",
                unsafe_allow_html=True,
            )

elif mode == "Real-Time Camera":
    # Start camera capture
    run_camera = st.checkbox("Start Camera")

    if run_camera:
        # Open webcam using OpenCV
        cap = cv2.VideoCapture(0)

        stframe = st.empty()  # Streamlit container for video frames

        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access the camera.")
                break
            coords = detect_face(frame)
            for x,y,w,h in coords:
                print(x,y,w,h)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
            # Preprocess the frame and predict emotion
            test_input = preprocess_frame(frame)
            prediction_array = model.predict(test_input)
            y_pred = ((prediction_array >= 0.5).astype(int)).item()
            face_label = mask_detect[y_pred]
            # Overlay the emotion on the video frame
            cv2.putText(
                frame,
                face_label,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Resize the frame for better display in Streamlit
            resized_frame = cv2.resize(frame, (600, 400))  # Resize to 640x480

            # Display the frame in the Streamlit app
            stframe.image(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB), use_container_width=False)

        cap.release()  # Release the webcam when the loop ends
