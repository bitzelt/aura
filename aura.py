import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(page_title="Aura", layout="wide", initial_sidebar_state="expanded")

# ── CSS global ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Yatra+One&display=swap');

    .stApp { background-color: #FFFFFF !important; }
    #MainMenu, footer, header { visibility: hidden; }

    [data-testid="stSidebar"] {
        background-color: #111111 !important;
    }

    .titulo-india {
        font-family: 'Yatra One', cursive;
        font-size: 50px;
        color: #ffffff;
        text-align: center;
        margin-top: 20px;
        letter-spacing: 2px;
    }

    .sidebar-label {
        font-family: 'Yatra One', cursive;
        color: #cccccc;
        font-size: 14px;
        margin-top: 16px;
    }

    /* Ocultar labels nativos de sliders */
    [data-testid="stSidebar"] .stSlider label { display: none; }

    .video-wrapper {
        background-color: #1a1a1a;
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.25);
        max-width: 860px;
        margin: 20px auto;
    }

    /* Pantalla completa con F11 */
    body.fs-mode [data-testid="stSidebar"] { display: none !important; }
    body.fs-mode .video-wrapper {
        position: fixed;
        inset: 0;
        max-width: 100vw;
        width: 100vw;
        height: 100vh;
        margin: 0;
        border-radius: 0;
        padding: 0;
        z-index: 99999;
        background: #000;
    }
</style>

<script>
(function() {
    const doc = window.parent.document;
    function applyFS() {
        doc.body.classList.toggle('fs-mode', !!doc.fullscreenElement);
    }
    doc.addEventListener('fullscreenchange', applyFS);
    doc.addEventListener('keydown', (e) => {
        if (e.key === 'F11') setTimeout(applyFS, 150);
    });
})();
</script>
""", unsafe_allow_html=True)

# ── Cargar modelo (cacheado) ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("yolo26n-seg.pt")

# ── Colores Chakras (BGR) ─────────────────────────────────────────────────────
CHAKRA_BGR = {
    0: (0,   0,   255),
    1: (0,   127, 255),
    2: (0,   255, 255),
    3: (0,   255, 0  ),
    4: (255, 191, 0  ),
    5: (130, 0,   75 ),
    6: (211, 0,   148),
}

# ── Procesador de video para webrtc ──────────────────────────────────────────
class AuraProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model()
        self.intensity = 0.55
        self.expand    = 2
        self.blur      = 65

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        results = self.model.track(img, classes=[0], persist=True, verbose=False)
        aura_mask = np.zeros_like(img)

        if results[0].masks is not None and results[0].boxes.id is not None:
            masks     = results[0].masks.data
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for mask, track_id in zip(masks, track_ids):
                mask_np = cv2.resize(
                    mask.cpu().numpy(), (w, h)
                ).astype(np.uint8)
                color_bgr = CHAKRA_BGR[track_id % 7]
                colored = np.zeros_like(img)
                colored[mask_np == 1] = color_bgr
                aura_mask = cv2.add(aura_mask, colored)

            kernel = np.ones((21, 21), np.uint8)
            aura_mask = cv2.dilate(aura_mask, kernel, iterations=self.expand)
            blur_k = self.blur if self.blur % 2 == 1 else self.blur + 1
            aura_mask = cv2.GaussianBlur(aura_mask, (blur_k, blur_k), 0)

        out = cv2.addWeighted(img, 1.0, aura_mask, self.intensity, 0)
        return av.VideoFrame.from_ndarray(out, format="bgr24")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="titulo-india">Aura</p>', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-label">Intensidad del aura</p>', unsafe_allow_html=True)
    aura_intensity = st.slider("Intensidad", 0.1, 1.0, 0.55, 0.05, label_visibility="hidden")

    st.markdown('<p class="sidebar-label">Expansión del aura</p>', unsafe_allow_html=True)
    aura_expand = st.slider("Expansión", 1, 5, 2, 1, label_visibility="hidden")

    st.markdown('<p class="sidebar-label">Suavizado</p>', unsafe_allow_html=True)
    aura_blur = st.slider("Suavizado", 11, 99, 65, 2, label_visibility="hidden")

    st.markdown("---")
    st.markdown(
        '<p style="color:#555;font-size:11px;margin-top:8px;">F11 → pantalla completa</p>',
        unsafe_allow_html=True
    )

# ── Área de video ─────────────────────────────────────────────────────────────
st.markdown('<div class="video-wrapper">', unsafe_allow_html=True)

ctx = webrtc_streamer(
    key="aura",
    video_processor_factory=AuraProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Actualizar parámetros del procesador en tiempo real
if ctx.video_processor:
    ctx.video_processor.intensity = aura_intensity
    ctx.video_processor.expand    = aura_expand
    ctx.video_processor.blur      = aura_blur

st.markdown('</div>', unsafe_allow_html=True)
