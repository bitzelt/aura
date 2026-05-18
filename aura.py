import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

st.set_page_config(page_title="Aura", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Yatra+One&display=swap');

    .stApp { background-color: #ffffff !important; }
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="collapsedControl"] { display: none; }
    section[data-testid="stSidebar"] { display: none; }

    .aura-logo {
        font-family: 'Yatra One', cursive;
        font-size: 32px;
        color: #222;
        text-align: left;
        padding: 16px 0 8px 0;
        letter-spacing: 3px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="aura-logo">Aura</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("yolo26n-seg.pt")

CHAKRA_BGR = [
    (0,   0,   220),
    (0,   110, 255),
    (0,   220, 255),
    (0,   200, 0  ),
    (255, 160, 0  ),
    (180, 0,   80 ),
    (220, 0,   180),
]

class AuraProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        try:
            results = self.model.track(img, classes=[0], persist=True, verbose=False)
            aura_mask = np.zeros_like(img)

            if results[0].masks is not None and results[0].boxes.id is not None:
                masks     = results[0].masks.data
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for mask, track_id in zip(masks, track_ids):
                    mask_np = cv2.resize(mask.cpu().numpy(), (w, h)).astype(np.uint8)
                    colored = np.zeros_like(img)
                    colored[mask_np == 1] = CHAKRA_BGR[track_id % 7]
                    aura_mask = cv2.add(aura_mask, colored)

                kernel    = np.ones((21, 21), np.uint8)
                aura_mask = cv2.dilate(aura_mask, kernel, iterations=2)
                aura_mask = cv2.GaussianBlur(aura_mask, (65, 65), 0)
                img       = cv2.addWeighted(img, 1.0, aura_mask, 0.55, 0)

        except Exception:
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="aura",
    video_processor_factory=AuraProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": {"width": 854, "height": 480}, "audio": False},
    async_processing=False,
)

st.markdown(
    '<p style="text-align:center;color:#bbb;font-size:12px;margin-top:8px;font-family:monospace;">F11 · pantalla completa</p>',
    unsafe_allow_html=True
)
