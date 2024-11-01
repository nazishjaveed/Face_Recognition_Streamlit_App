import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title('Face Recognition App')

@st.cache_resource
def load_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_cascade = load_cascade()

def detect_faces(image):
    image_np = np.array(image)  # Convert PIL Image to NumPy array
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(image_np, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image_np, faces

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file:
    image = Image.open(file)
    result_img, result_faces = detect_faces(image)
    st.image(result_img, caption='Processed Image', use_column_width=True)
    st.success(f'Found {len(result_faces)} face(s)')
