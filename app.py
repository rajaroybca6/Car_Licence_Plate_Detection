import os
import streamlit as st
import cv2
import av
import logging
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from plate_finder import PlateFinder
from ocr import OCR

# Set up logging to see errors in the Streamlit Sidebar/Logs
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="License Plate Recognition",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🚗 License Plate Recognition")
st.caption("Optimized for Mobile Data (4G/5G) via TURN Relay")

# -------------------------------------------------------------------
# THE MOBILE DATA FIX (STRICT TURN CONFIG)
# -------------------------------------------------------------------

# Get secrets from Streamlit Cloud
turn_url_1 = os.getenv("TURN_URL_1", "turn:openrelay.metered.ca:80")
turn_url_2 = os.getenv("TURN_URL_2", "turn:openrelay.metered.ca:443?transport=tcp")
turn_url_3 = os.getenv("TURN_URL_3", "turns:openrelay.metered.ca:443?transport=tcp")
turn_username = os.getenv("TURN_USERNAME", "openrelayproject")
turn_password = os.getenv("TURN_PASSWORD", "openrelayproject")

# We force the connection to use the TURN server immediately
# This bypasses the mobile carrier's P2P firewall.
RTC_CONFIG = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": [turn_url_1, turn_url_2, turn_url_3],
                "username": turn_username,
                "credential": turn_password,
            },
        ],
        "iceTransportPolicy": "relay",  # CRITICAL: Forces the use of the TURN server
    }
)


# -------------------------------------------------------------------
# Model Loading
# -------------------------------------------------------------------

@st.cache_resource
def load_models():
    try:
        finder = PlateFinder(minPlateArea=4100, maxPlateArea=15000)
        ocr = OCR(
            modelFile="model/binary_128_0.50_ver3.pb",
            labelFile="model/binary_128_0.50_labels_ver2.txt"
        )
        return finder, ocr
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None


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

        # Mobile Optimization: Resize to 640px width
        h, w = img.shape[:2]
        if w > 640:
            scale = 640 / w
            img = cv2.resize(img, (640, int(h * scale)))

        # Process every 5th frame to keep the mobile CPU cool
        if self.frame_counter % 5 == 0:
            self.current_plates = []
            self.plate_detected = False

            try:
                possible_plates = plate_finder.find_possible_plates(img)

                if possible_plates:
                    for i, plate_img in enumerate(possible_plates):
                        if i >= len(plate_finder.char_on_plate): continue

                        chars = plate_finder.char_on_plate[i]
                        text, count = ocr_model.label_image_list(chars, image_size=128)

                        if count > 0 and text.strip():
                            clean_text = text.strip()
                            self.current_plates.append(clean_text)
                            self.plate_detected = True

                            # Draw on image
                            if hasattr(plate_finder, "plate_locations") and i < len(plate_finder.plate_locations):
                                x, y, w_p, h_p = plate_finder.plate_locations[i]
                                cv2.rectangle(img, (x, y), (x + w_p, y + h_p), (0, 255, 0), 2)
                                cv2.putText(img, clean_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Status Overlay
                color = (0, 255, 0) if self.plate_detected else (0, 0, 255)
                msg = "DETECTED" if self.plate_detected else "SCANNING..."
                cv2.putText(img, msg, (10, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            except Exception as e:
                logger.error(f"Processing error: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -------------------------------------------------------------------
# UI Layout
# -------------------------------------------------------------------

camera_facing = st.radio("Select Camera", ["Back (Rear)", "Front (Selfie)"], horizontal=True)
facing_mode = "environment" if camera_facing == "Back (Rear)" else "user"

ctx = webrtc_streamer(
    key=f"plate-reader-{facing_mode}",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={
        "video": {"facingMode": {"ideal": facing_mode}},
        "audio": False,
    },
    async_processing=True,
)

st.markdown("---")

# Show Results Below Camera
if ctx.video_processor:
    plates = ctx.video_processor.current_plates
    if plates:
        for p in plates:
            if p not in st.session_state.plates_found:
                st.session_state.plates_found.append(p)
            st.success(f"Plate Found: **{p}**")
    else:
        st.info("No plates currently in view.")

with st.expander("Session History"):
    for plate in reversed(st.session_state.plates_found):
        st.write(f"- `{plate}`")