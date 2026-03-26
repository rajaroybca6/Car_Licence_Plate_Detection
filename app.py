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

st.title("🚗 License Plate Recognition")
st.caption("Detects multiple plates at once — works on Wi-Fi and Mobile Data.")

# --- THIS IS THE ONLY PART UPDATED TO ALLOW MOBILE DATA ---
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        # TURN servers bypass cellular firewalls (Mobile Data)
        {
            "urls": ["turn:openrelay.metered.ca:80", "turn:openrelay.metered.ca:443"],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        },
        {
            "urls": ["turns:openrelay.metered.ca:443?transport=tcp"],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        }
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

if "plates_found" not in st.session_state:
    st.session_state.plates_found = []

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_counter = 0
        self.current_plates = []   # ← list for multiple plates
        self.plate_detected = False  # ← YES/NO flag

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_counter += 1

        h, w = img.shape[:2]
        if w > 640:
            scale = 640 / w
            img = cv2.resize(img, (640, int(h * scale)))

        if self.frame_counter % 4 == 0:
            # ── Reset every frame ──────────────────────────────────────────
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
                            self.current_plates.append(text)  # ← add each plate
                            self.plate_detected = True         # ← YES

                            # Draw box per plate
                            if (
                                hasattr(plate_finder, "plate_locations")
                                and i < len(plate_finder.plate_locations)
                            ):
                                x, y, w_p, h_p = plate_finder.plate_locations[i]
                                # Different color per plate
                                colors = [
                                    (0, 255, 0),    # green
                                    (0, 165, 255),  # orange
                                    (255, 0, 0),    # blue
                                    (0, 0, 255),    # red
                                    (255, 255, 0),  # cyan
                                ]
                                color = colors[i % len(colors)]
                                cv2.rectangle(
                                    img, (x, y), (x + w_p, y + h_p),
                                    color, 2
                                )
                                cv2.putText(
                                    img, f"#{i+1}: {text}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    color, 2
                                )
                            else:
                                cv2.putText(
                                    img, f"#{i+1}: {text}",
                                    (20, 40 + i * 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                    (0, 255, 0), 2
                                )

                # ── Show YES/NO banner on video ────────────────────────────
                if self.plate_detected:
                    cv2.putText(
                        img, f"✓ {len(self.current_plates)} PLATE(S) DETECTED",
                        (10, img.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2
                    )
                else:
                    cv2.putText(
                        img, "✗ NO PLATE DETECTED",
                        (10, img.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2
                    )

            except Exception:
                cv2.putText(
                    img, "Processing...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 165, 255), 2
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ── Camera selector ───────────────────────────────────────────────────────────
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

# ── Live Status Panel ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔍 Live Detection Status")

if ctx.video_processor:
    plates = ctx.video_processor.current_plates
    detected = ctx.video_processor.plate_detected

    if detected and plates:
        # ── YES — show each plate ──────────────────────────────────────────
        st.success(f"✅ YES — {len(plates)} Plate(s) Detected")
        for i, plate in enumerate(plates):
            st.markdown(f"**Car #{i+1}** → `{plate}`")

            # Save to history
            if plate not in st.session_state.plates_found:
                st.session_state.plates_found.append(plate)
    else:
        # ── NO — no plate in view ──────────────────────────────────────────
        st.error("❌ NO — No Plate In View")

else:
    st.info("📷 Click START to begin")

# ── Detection History ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🗂️ Detection History (All Session)")

if st.session_state.plates_found:
    st.markdown(f"**Total unique plates detected: {len(st.session_state.plates_found)}**")
    for i, plate in enumerate(reversed(st.session_state.plates_found)):
        st.write(f"**{i+1}.** `{plate}`")

    if st.button("🗑️ Clear History"):
        st.session_state.plates_found = []
        st.rerun()
else:
    st.info("No plates recorded yet this session.")

# ── Tips ──────────────────────────────────────────────────────────────────────
with st.expander("💡 Tips for best results"):
    st.markdown("""
    - ✅ Works with **multiple cars** in frame simultaneously
    - ✅ Each plate gets a **different color box** (#1 green, #2 orange, #3 blue...)
    - ✅ Bottom of video shows **YES/NO** at all times
    - ✅ Good lighting helps accuracy
    - ✅ Hold camera **30–60cm** from plate
    - ⚠️ iPhone users: use **Safari**
    """)