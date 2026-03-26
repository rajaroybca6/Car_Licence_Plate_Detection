import streamlit as st
import cv2
from plate_finder import PlateFinder
from ocr import OCR

st.title("License Plate Recognition (Auto Webcam)")

# Initialize
plate_finder = PlateFinder(minPlateArea=4100, maxPlateArea=15000)
ocr_model = OCR(
    modelFile="model/binary_128_0.50_ver3.pb",
    labelFile="model/binary_128_0.50_labels_ver2.txt"
)

# AUTO OPEN WEBCAM
cap = cv2.VideoCapture(0)

frame_window = st.image([])

# Run limited frames to avoid freeze
for frame_count in range(300):  # auto-run ~300 frames
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not working")
        break

    possible_plates = plate_finder.find_possible_plates(frame)

    if possible_plates is not None:
        for i, plate_img in enumerate(possible_plates):
            chars = plate_finder.char_on_plate[i]
            text, count = ocr_model.label_image_list(chars, image_size=128)

            if count > 0:
                st.write(f"Plate: {text}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame)

cap.release()

    #streamlit run app.py