import os
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

st.title("⚖️ **GDPR / Privacy Notice**\n\n"
    "This is a technical Proof of Concept (PoC) created exclusively for "
    "**school/college and study purposes**. \n\n"
    "**Data Handling:** No data is stored permanently. Video frames and "
    "license plate strings are processed in-memory in real-time and deleted "
    "upon session close. \n\n"
    "**Usage:** Please use only on your own vehicle or with explicit consent."
"This is only for Study Purposes.")
st.title("🚗 License Plate Recognition")
st.caption("Detects multiple plates at once — works on Wi-Fi and mobile networks.")

# -------------------------------------------------------------------
# WebRTC / TURN config — reads from Streamlit Secrets
# -------------------------------------------------------------------
def _get_config(key, default=""):
    return st.secrets.get(key, os.getenv(key, default))

turn_url_1    = _get_config("TURN_URL_1")
turn_url_2    = _get_config("TURN_URL_2")
turn_url_3    = _get_config("TURN_URL_3")
turn_url_4    = _get_config("TURN_URL_4")
turn_username = _get_config("TURN_USERNAME")
turn_password = _get_config("TURN_PASSWORD")
force_turn    = _get_config("FORCE_TURN", "false").lower() == "true"

# Build ICE server list
turn_urls = [u for u in [turn_url_1, turn_url_2, turn_url_3, turn_url_4] if u.strip()]

if turn_urls and turn_username and turn_password:
    ice_servers = [
        {"urls": ["stun:stun.relay.metered.ca:80"]},
        {
            "urls": turn_urls,
            "username": turn_username,
            "credential": turn_password,
        },
    ]
else:
    # Fallback: free public relay
    ice_servers = [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": [
                "turn:openrelay.metered.ca:80",
                "turn:openrelay.metered.ca:443",
                "turn:openrelay.metered.ca:443?transport=tcp",
            ],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
    ]

rtc_config_data = {"iceServers": ice_servers}

if force_turn:
    rtc_config_data["iceTransportPolicy"] = "relay"

RTC_CONFIG = RTCConfiguration(rtc_config_data)


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


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_counter = 0
        self.current_plates = []
        self.plate_detected = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_counter += 1

        h, w = img.shape[:2]
        if w > 640:
            scale = 640 / w
            img = cv2.resize(img, (640, int(h * scale)))

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

                            if (
                                hasattr(plate_finder, "plate_locations")
                                and i < len(plate_finder.plate_locations)
                            ):
                                x, y, w_p, h_p = plate_finder.plate_locations[i]
                                colors = [
                                    (0, 255, 0),
                                    (0, 165, 255),
                                    (255, 0, 0),
                                    (0, 0, 255),
                                    (255, 255, 0),
                                ]
                                color = colors[i % len(colors)]

                                cv2.rectangle(
                                    img,
                                    (x, y),
                                    (x + w_p, y + h_p),
                                    color,
                                    2
                                )
                                cv2.putText(
                                    img,
                                    f"#{i+1}: {text}",
                                    (x, max(y - 10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    color,
                                    2
                                )
                            else:
                                cv2.putText(
                                    img,
                                    f"#{i+1}: {text}",
                                    (20, 40 + i * 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9,
                                    (0, 255, 0),
                                    2
                                )

                if self.plate_detected:
                    cv2.putText(
                        img,
                        f"YES - {len(self.current_plates)} PLATE(S) DETECTED",
                        (10, img.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                else:
                    cv2.putText(
                        img,
                        "NO PLATE DETECTED",
                        (10, img.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

            except Exception:
                cv2.putText(
                    img,
                    "Processing...",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    2
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


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
                "facingMode": {"ideal": facing_mode},
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
            },
            "audio": False,
        },
        async_processing=True,
    )

st.markdown("---")
st.markdown("### Live Detection Status")

if ctx.video_processor:
    plates = ctx.video_processor.current_plates
    detected = ctx.video_processor.plate_detected

    if detected and plates:
        st.success(f"✅ YES - {len(plates)} Plate(s) Detected")
        for i, plate in enumerate(plates):
            st.markdown(f"**Car #{i+1}** → `{plate}`")
            if plate not in st.session_state.plates_found:
                st.session_state.plates_found.append(plate)
    else:
        st.error("❌ NO - No Plate In View")
else:
    st.info("📷 Click START to begin")

st.markdown("---")
st.markdown("### Detection History (All Session)")

if st.session_state.plates_found:
    st.markdown(
        f"**Total unique plates detected: {len(st.session_state.plates_found)}**"
    )
    for i, plate in enumerate(reversed(st.session_state.plates_found)):
        st.write(f"**{i+1}.** `{plate}`")

    if st.button("🗑️ Clear History"):
        st.session_state.plates_found = []
        st.rerun()
else:
    st.info("No plates recorded yet this session.")

with st.expander("💡 Tips for best results"):
    st.markdown("""
    - ✅ Works with **multiple cars** in frame simultaneously
    - ✅ Each plate gets a **different colour box**
    - ✅ Now works on **Wi-Fi and mobile networks** (4G/5G)
    - ✅ Good lighting improves accuracy
    - ✅ Hold camera **30–60 cm** from plate
    - ⚠️ iPhone users: use **Safari**
    - ⚠️ Must be deployed on **public HTTPS** for mobile camera access
    -"⚖️ **GDPR / Privacy Notice**\n\n"
    "This is a technical Proof of Concept (PoC) created exclusively for "
    "**school/college and study purposes**. \n\n"
    "**Data Handling:** No data is stored permanently. Video frames and "
    "license plate strings are processed in-memory in real-time and deleted "
    "upon session close. \n\n"
    "**Usage:** Please use only on your own vehicle or with explicit consent."
"This is only for Study Purposes".
    """)
st.sidebar.markdown("---")
st.sidebar.info(
    "⚖️ **GDPR / Privacy Notice**\n\n"
    "This is a technical Proof of Concept (PoC) created exclusively for "
    "**school/college and study purposes**. \n\n"
    "**Data Handling:** No data is stored permanently. Video frames and "
    "license plate strings are processed in-memory in real-time and deleted "
    "upon session close. \n\n"
    "**Usage:** Please use only on your own vehicle or with explicit consent."
"This is only for Study Purposes.")
# streamlit run app.py
