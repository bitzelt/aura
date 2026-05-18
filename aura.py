import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
# Se usa VideoTransformerBase que es el estándar actual
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

st.set_page_config(page_title="Aura", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Yatra+One&display=swap');
    .stApp { background-color: #ffffff !important; }
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="collapsedControl"] { display: none; }
    section[data-testid="stSidebar"] { display: none; }

    #video-wrap {
        position: relative;
        display: inline-block;
        width: 100%;
    }
    #fs-btn {
        position: absolute;
        top: 10px;
        right: 10px;
        z-index: 9999;
        background: rgba(0,0,0,0.45);
        color: #fff;
        border: none;
        border-radius: 5px;
        padding: 5px 10px;
        font-size: 16px;
        cursor: pointer;
        backdrop-filter: blur(4px);
    }
    #fs-btn:hover { background: rgba(0,0,0,0.75); }
    #video-wrap:fullscreen,
    #video-wrap:-webkit-full-screen {
        background: #000;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100vw;
        height: 100vh;
    }
    #video-wrap:fullscreen video,
    #video-wrap:fullscreen canvas,
    #video-wrap:-webkit-full-screen video,
    #video-wrap:-webkit-full-screen canvas {
        width: 100vw !important;
        height: 100vh !important;
        object-fit: contain;
    }
</style>

<script>
function setupFS() {
    const wrap = window.parent.document.getElementById('video-wrap');
    const btn  = window.parent.document.getElementById('fs-btn');
    if (!wrap || !btn) { setTimeout(setupFS, 500); return; }
    btn.addEventListener('click', () => {
        if (!window.parent.document.fullscreenElement) {
            wrap.requestFullscreen().catch(() => {});
            btn.textContent = '✕';
        } else {
            window.parent.document.exitFullscreen();
            btn.textContent = '⛶';
        }
    });
    window.parent.document.addEventListener('fullscreenchange', () => {
        if (!window.parent.document.fullscreenElement) btn.textContent = '⛶';
    });
}
setupFS();
</script>
""", unsafe_allow_html=True)

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

class AuraProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = load_model()

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        try:
            results = self.model.predict(img, classes=[0], verbose=False)
            aura_mask = np.zeros_like(img)
            if results[0].masks is not None:
                for i, mask in enumerate(results[0].masks.data):
                    mask_np = cv2.resize(mask.cpu().numpy(), (w, h)).astype(np.uint8)
                    colored = np.zeros_like(img)
                    colored[mask_np == 1] = CHAKRA_BGR[i % 7]
                    aura_mask = cv2.add(aura_mask, colored)
                kernel    = np.ones((21, 21), np.uint8)
                aura_mask = cv2.dilate(aura_mask, kernel, iterations=2)
                aura_mask = cv2.GaussianBlur(aura_mask, (65, 65), 0)
                img       = cv2.addWeighted(img, 1.0, aura_mask, 0.55, 0)
        except Exception:
            pass
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.markdown('<div id="video-wrap"><button id="fs-btn">⛶</button>', unsafe_allow_html=True)

webrtc_streamer(
    key="aura",
    video_transformer_factory=AuraProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": {"width": 854, "height": 480}, "audio": False},
    async_processing=True,
)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:#bbb;font-size:12px;margin-top:8px;font-family:monospace;">⛶ pantalla completa</p>',
    unsafe_allow_html=True
)
