import os
import streamlit as st
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from plate_finder import PlateFinder
from ocr import OCR

# 1. Page Config
st.set_page_config(
    page_title="License Plate Recognition",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for mobile responsiveness
st.markdown("""
    <style>
        .main { padding: 1rem; }
        h1 { font-size: 1.5rem !important; }
        @media (max-width: 768px) {
            .main { padding: 0.5rem; }
            h1 { font-size: 1.2rem !important; }
        }
    </style>
""", unsafe_allow_html=True)

st.title("License Plate Recognition")
st.caption("Works on Wi-Fi and Mobile Data (4G/5G).")

# -------------------------------------------------------------------
# WebRTC Configuration (Pulling from Streamlit Secrets)
# -------------------------------------------------------------------

# We use str() to ensure the values are read correctly from Secrets
turn_url_1 = str(os.getenv("TURN_URL_1", ""))
turn_url_2 = str(os.getenv("TURN_URL_2", ""))
turn_url_3 = str(os.getenv("TURN_URL_3", ""))
turn_username = str(os.getenv("TURN_USERNAME", ""))
turn_password = str(os.getenv("TURN_PASSWORD", ""))
force_turn = str(os.getenv("FORCE_TURN", "false")).lower() == "true"

# STUN servers are free and work for Wi-Fi
ice_servers = [
    {"urls": ["stun:stun.l.google.com:19302"]},
    {"urls": ["stun:stun1.l.google.com:19302"]},
]

# Add the TURN servers for Mobile Data
turn_urls = [url for url in [turn_url_1, turn_url_2, turn_url_3] if url.strip()]
if turn_urls and turn_username and turn_password:
    ice_servers.append(
        {
            "urls": turn_urls,
            "username": turn_username,
            "credential": turn_password,
        }
    )

# Force the connection to use the TURN relay if on mobile
rtc_config_data = {
    "iceServers": ice_servers,
}

if force_turn:
    # This is the "Magic" line that fixes mobile data connection
    rtc_config_data["iceTransportPolicy"] = "relay"

RTC_CONFIG = RTCConfiguration(rtc_config_data)

# -------------------------------------------------------------------
# AI Model Loading
# -------------------------------------------------------------------

@st.cache_resource
def load_models():
    finder = PlateFinder(minPlateArea=4100, maxPlateArea=15000)
    ocr = OCR(
        modelFile="model/binary_128_0.50_ver3.pb",
        labelFile="model/binary_128_0.50_labels_ver2.txt"
    )
    return finder, ocr

plate_finder, ocr_model = load_models()

if "plates_found" not in st.session_state:
    st.session_state.plates_found = []

# -------------------------------------------------------------------
# Video Processing Logic
# -------------------------------------------------------------------

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_counter = 0
        self.current_plates = []
        self.plate_detected = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_counter += 1

        # Resize for performance on mobile
        h, w = img.shape[:2]
        if w > 640:
            scale = 640 / w
            img = cv2.resize(img, (640, int(h * scale)))

        # Process every 4th frame to reduce CPU lag
        if self.frame_counter % 4 == 0:
            self.current_plates = []
            self.plate_detected = False

            try:
                possible_plates = plate_finder.find_possible_plates(img)

                if possible_plates:
                    for i, plate_img in enumerate(possible_plates):
                        if i >= len(plate_finder.char_on_plate):
                            continue

                        chars = plate_finder.char_on_plate[i]

                        try:
                            text, count = ocr_model.label_image_list(
                                chars, image_size=128
                            )
                        except Exception:
                            continue

                        if count > 0 and text.strip():
                            text = text.strip()
                            self.current_plates.append(text)
                            self.plate_detected = True

                            if hasattr(plate_finder, "plate_locations") and i < len(plate_finder.plate_locations):
                                x, y, w_p, h_p = plate_finder.plate_locations[i]
                                colors = [(0, 255, 0), (0, 165, 255), (255, 0, 0), (0, 0, 255)]
                                color = colors[i % len(colors)]

                                cv2.rectangle(img, (x, y), (x + w_p, y + h_p), color, 2)
                                cv2.putText(img, f"#{i+1}: {text}", (x, max(y - 10, 20)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Feedback overlay
                status_color = (0, 255, 0) if self.plate_detected else (0, 0, 255)
                status_text = f"YES - {len(self.current_plates)} DETECTED" if self.plate_detected else "NO PLATE"
                cv2.putText(img, status_text, (10, img.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            except Exception:
                pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -------------------------------------------------------------------
# User Interface
# -------------------------------------------------------------------

col1, col2 = st.columns([3, 1])

with col2:
    camera_facing = st.selectbox("Camera", ["Back (Rear)", "Front (Selfie)"], index=0)

facing_mode = "environment" if camera_facing == "Back (Rear)" else "user"

with col1:
    ctx = webrtc_streamer(
        key=f"plate-reader-{facing_mode}",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={
            "video": {
                "facingMode": {"ideal": facing_mode},
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
            },
            "audio": False,
        },
        async_processing=True,
    )

st.markdown("---")

# Live Update Section
if ctx.video_processor:
    plates = ctx.video_processor.current_plates
    if plates:
        st.success(f"Detected: {', '.join(plates)}")
        for plate in plates:
            if plate not in st.session_state.plates_found:
                st.session_state.plates_found.append(plate)
    else:
        st.warning("Scanning for plates...")

# History Section
with st.expander("Detection History"):
    if st.session_state.plates_found:
        for i, plate in enumerate(reversed(st.session_state.plates_found)):
            st.write(f"**{i+1}.** `{plate}`")
        if st.button("Clear History"):
            st.session_state.plates_found = []
            st.rerun()
    else:
        st.info("No plates saved yet.")