import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

st.set_page_config(page_title="Aura", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@700&display=swap');

    .stApp { background-color: #0d0d0d !important; }
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="collapsedControl"] { display: none; }
    section[data-testid="stSidebar"] { display: none; }

    .aura-logo {
        font-family: 'Cinzel Decorative', cursive;
        font-size: 28px;
        color: #e8d5ff;
        letter-spacing: 6px;
        padding: 18px 0 0 28px;
        text-shadow: 0 0 20px rgba(180,100,255,0.6);
    }

    .video-frame {
        display: block;
        margin: 10px auto 0 auto;
        width: 70vw;
        max-width: 900px;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 0 60px rgba(150, 80, 255, 0.25), 0 4px 30px rgba(0,0,0,0.8);
        background: #111;
    }

    .video-frame > div,
    .video-frame video,
    .video-frame canvas {
        width: 100% !important;
        border-radius: 10px;
    }

    .video-frame button {
        background: #7c3aed !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 10px 24px !important;
        font-size: 15px !important;
        cursor: pointer !important;
        margin: 16px auto !important;
        display: block !important;
    }

    .fullscreen-hint {
        text-align: center;
        color: #444;
        font-size: 12px;
        letter-spacing: 2px;
        margin-top: 14px;
        font-family: monospace;
    }

    .chakra-info {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-top: 12px;
        flex-wrap: wrap;
    }
    .chakra-pill {
        font-size: 11px;
        padding: 3px 10px;
        border-radius: 20px;
        letter-spacing: 1px;
        font-family: monospace;
        opacity: 0.7;
    }
</style>

<script>
(function() {
    const doc = window.parent.document;
    doc.addEventListener('keydown', (e) => {
        if (e.key === 'F11') {
            e.preventDefault();
            if (!doc.fullscreenElement) {
                doc.documentElement.requestFullscreen();
            } else {
                doc.exitFullscreen();
            }
        }
    });
})();
</script>
""", unsafe_allow_html=True)

st.markdown('<div class="aura-logo">✦ AURA</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("yolo26n-seg.pt")

CHAKRAS = [
    {"name": "Muladhara",     "bgr": (0,   0,   220), "hex": "#dc0000"},
    {"name": "Svadhisthana",  "bgr": (0,   110, 255), "hex": "#ff6e00"},
    {"name": "Manipura",      "bgr": (0,   220, 255), "hex": "#ffdc00"},
    {"name": "Anahata",       "bgr": (0,   200, 0  ), "hex": "#00c800"},
    {"name": "Vishuddha",     "bgr": (255, 160, 0  ), "hex": "#00a0ff"},
    {"name": "Ajna",          "bgr": (180, 0,   80 ), "hex": "#5000b4"},
    {"name": "Sahasrara",     "bgr": (220, 0,   180), "hex": "#b400dc"},
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
                    mask_np = cv2.resize(
                        mask.cpu().numpy(), (w, h)
                    ).astype(np.uint8)

                    chakra  = CHAKRAS[track_id % 7]
                    colored = np.zeros_like(img)
                    colored[mask_np == 1] = chakra["bgr"]
                    aura_mask = cv2.add(aura_mask, colored)

                    # Nombre del chakra sobre la persona
                    ys, xs = np.where(mask_np == 1)
                    if len(xs) > 0:
                        cx, cy = int(xs.mean()), int(ys.min()) - 12
                        cy = max(cy, 20)
                        label = chakra["name"]
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                        cv2.rectangle(img,
                                      (cx - tw//2 - 6, cy - th - 6),
                                      (cx + tw//2 + 6, cy + 4),
                                      (0, 0, 0), -1)
                        cv2.putText(img, label, (cx - tw//2, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    chakra["bgr"], 1, cv2.LINE_AA)

                kernel    = np.ones((21, 21), np.uint8)
                aura_mask = cv2.dilate(aura_mask, kernel, iterations=2)
                aura_mask = cv2.GaussianBlur(aura_mask, (65, 65), 0)

            img = cv2.addWeighted(img, 1.0, aura_mask, 0.55, 0)

        except Exception:
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.markdown('<div class="video-frame">', unsafe_allow_html=True)

webrtc_streamer(
    key="aura",
    video_processor_factory=AuraProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
    async_processing=True,
)

st.markdown('</div>', unsafe_allow_html=True)

pills = "".join([
    f'<span class="chakra-pill" style="background:{c["hex"]}22; color:{c["hex"]}; border:1px solid {c["hex"]}55">{c["name"]}</span>'
    for c in CHAKRAS
])
st.markdown(f'<div class="chakra-info">{pills}</div>', unsafe_allow_html=True)

st.markdown('<div class="fullscreen-hint">F11 · PANTALLA COMPLETA</div>', unsafe_allow_html=True)
