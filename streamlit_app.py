import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np

FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
EYE_CASCADE = cv2.CascadeClassifier('haarcascade_eye.xml')
SMILE_CASCADE = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect_faces(image):
    new_img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(gray, 1.3,1)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 5)
        roi_gray = gray[y:y+h, x:x+h]
        roi_color = img[y:y+h, x:x+h]
        eyes = EYE_CASCADE.detectMultiScale(roi_gray)
        smiles = SMILE_CASCADE.detectMultiScale(roi_gray, 2, 4)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), ((sx+sw), (sy+sh)), (0,0,255), 2)

    return img, faces

def main():
    st.title("Project Face Detection")
    
    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("Pilih", activities)

    if choice == "Home":
        home()

    elif choice == "About":
        about()


def home():
    st.subheader("Face Detection")
    image_files = st.file_uploader("Upload foto: ", type=['jpg', 'png', 'jpeg'])

    if image_files is not None:
        image = Image.open(image_files)
        st.text("Original Image")
        st.image(image)

    task = ["Face Detection", "Face Recognition"]
    features = st.sidebar.selectbox("Task:", task)
    if st.button("Process"):
        if features == "Face Detection":
            result_img, result_face = detect_faces(image)
            st.success("Found {} faces". format(len(result_face)))
            st.image(result_img)

def about():
    st.title("About")
    st.header("Tentang sayan")
    st.text("lorem ipsum dolor sit amet")

if __name__ == "__main__":
    main()