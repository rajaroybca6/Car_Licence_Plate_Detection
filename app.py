import streamlit as st
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from plate_finder import PlateFinder
from ocr import OCR

st.set_page_config(
    page_title="License Plate Recognition",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS for mobile responsiveness
st.markdown("""
    <style>
        .main { padding: 1rem; }
        h1 { font-size: 1.5rem !important; }
        .stAlert { font-size: 0.9rem; }
        @media (max-width: 768px) {
            .main { padding: 0.5rem; }
            h1 { font-size: 1.2rem !important; }
        }
    </style>
""", unsafe_allow_html=True)

st.title("🚗 License Plate Recognition")
st.caption("Works on mobile, tablet, and desktop — point your camera at a license plate.")

# ── ICE / STUN servers so WebRTC works across all networks & devices ──────────
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]
})

@st.cache_resource
def load_models():
    finder = PlateFinder(minPlateArea=4100, maxPlateArea=15000)
    ocr = OCR(
        modelFile="model/binary_128_0.50_ver3.pb",
        labelFile="model/binary_128_0.50_labels_ver2.txt"
    )
    return finder, ocr

plate_finder, ocr_model = load_models()

# ── Detected plates display (uses session state to persist across frames) ─────
if "plates_found" not in st.session_state:
    st.session_state.plates_found = []

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_counter = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_counter += 1

        # Resize for faster processing on all devices
        h, w = img.shape[:2]
        if w > 640:
            scale = 640 / w
            img = cv2.resize(img, (640, int(h * scale)))

        # Process every 4th frame (balance speed vs accuracy)
        if self.frame_counter % 4 == 0:
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
                            # Draw green box + label on frame
                            if (
                                hasattr(plate_finder, "plate_locations")
                                and i < len(plate_finder.plate_locations)
                            ):
                                x, y, w_p, h_p = plate_finder.plate_locations[i]
                                cv2.rectangle(
                                    img, (x, y), (x + w_p, y + h_p),
                                    (0, 255, 0), 2
                                )
                                cv2.putText(
                                    img, text, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (0, 255, 0), 2
                                )
                            else:
                                cv2.putText(
                                    img, f"Plate: {text}",
                                    (20, 40 + i * 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                    (0, 255, 0), 2
                                )

            except Exception as e:
                cv2.putText(
                    img, "Processing...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ── Camera selector (front / rear for mobile) ─────────────────────────────────
col1, col2 = st.columns([3, 1])
with col2:
    camera_facing = st.selectbox(
        "Camera",
        ["Back (Rear)", "Front (Selfie)"],
        index=0
    )

facing_mode = "environment" if camera_facing == "Back (Rear)" else "user"

with col1:
    ctx = webrtc_streamer(
        key=f"plate-reader-{facing_mode}",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={
            "video": {
                "facingMode": facing_mode,
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
            },
            "audio": False,
        },
        async_processing=True,
    )

# ── Instructions ──────────────────────────────────────────────────────────────
with st.expander("ℹ️ How to use", expanded=False):
    st.markdown("""
    1. Click **START** above and allow camera access when prompted
    2. Point your camera at a **license plate**
    3. Hold steady — plates are detected automatically
    4. Use **Back (Rear)** camera on mobile for best results
    5. Good lighting improves accuracy
    """)

st.markdown("---")
st.caption("💡 Tip: On iPhone, use Chrome or Firefox for best WebRTC support.")
#streamlit run app.py