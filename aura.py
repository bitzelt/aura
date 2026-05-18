import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(page_title="Aura", layout="wide", initial_sidebar_state="expanded")

# ── CSS global ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Yatra+One&display=swap');

    /* Fondo blanco general */
    .stApp { background-color: #FFFFFF !important; }

    /* Ocultar elementos nativos de Streamlit */
    #MainMenu, footer, header { visibility: hidden; }

    /* Sidebar negro */
    [data-testid="stSidebar"] {
        background-color: #111111 !important;
    }

    /* Título Aura en sidebar */
    .titulo-india {
        font-family: 'Yatra One', cursive;
        font-size: 50px;
        color: #ffffff;
        text-align: center;
        margin-top: 20px;
        letter-spacing: 2px;
    }

    /* Label de controles en sidebar */
    .sidebar-label {
        font-family: 'Yatra One', cursive;
        color: #cccccc;
        font-size: 14px;
        margin-top: 16px;
    }

    /* Contenedor del video: fondo gris oscuro tipo monitor */
    .video-wrapper {
        background-color: #1a1a1a;
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.25);
        max-width: 860px;
        margin: 20px auto;
    }

    /* Modo pantalla completa al presionar F11 */
    body.fs-mode [data-testid="stSidebar"]   { display: none !important; }
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
    body.fs-mode .video-wrapper img {
        width: 100vw !important;
        height: 100vh !important;
        object-fit: contain;
    }
</style>

<script>
// Escuchar F11 y el evento nativo fullscreenchange
(function() {
    const doc = window.parent.document;

    function applyFullscreen() {
        if (doc.fullscreenElement) {
            doc.body.classList.add('fs-mode');
        } else {
            doc.body.classList.remove('fs-mode');
        }
    }

    doc.addEventListener('fullscreenchange', applyFullscreen);

    // F11 manual (algunos navegadores bloquean la API, esto cubre el CSS al menos)
    doc.addEventListener('keydown', (e) => {
        if (e.key === 'F11') {
            setTimeout(applyFullscreen, 150);
        }
    });
})();
</script>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="titulo-india">Aura</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label">Intensidad del aura</p>', unsafe_allow_html=True)
    aura_intensity = st.slider("", 0.1, 1.0, 0.55, 0.05, label_visibility="collapsed")
    st.markdown('<p class="sidebar-label">Expansión del aura</p>', unsafe_allow_html=True)
    aura_expand = st.slider(" ", 1, 5, 2, 1, label_visibility="collapsed")
    st.markdown('<p class="sidebar-label">Suavizado</p>', unsafe_allow_html=True)
    aura_blur = st.slider("  ", 11, 99, 65, 2, label_visibility="collapsed")

    st.markdown("---")
    run = st.toggle("▶  Activar cámara", value=True)
    st.markdown(
        '<p style="color:#555;font-size:11px;margin-top:24px;">Presioná F11 para pantalla completa</p>',
        unsafe_allow_html=True
    )

# ── Área de video ─────────────────────────────────────────────────────────────
st.markdown('<div class="video-wrapper">', unsafe_allow_html=True)
frame_slot = st.empty()
st.markdown('</div>', unsafe_allow_html=True)

# ── Cargar modelo ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # yolo26n-seg: modelo nano de segmentación de YOLO26 (se descarga automáticamente)
    return YOLO("yolo26n-seg.pt")

# ── Colores de los Chakras (BGR para OpenCV) ──────────────────────────────────
CHAKRA_BGR = {
    0: (0,   0,   255),   # Rojo     — Muladhara
    1: (0,   127, 255),   # Naranja  — Svadhisthana
    2: (0,   255, 255),   # Amarillo — Manipura
    3: (0,   255, 0  ),   # Verde    — Anahata
    4: (255, 191, 0  ),   # Celeste  — Vishuddha
    5: (130, 0,   75  ),  # Índigo   — Ajna
    6: (211, 0,   148),   # Violeta  — Sahasrara
}

# ── Loop principal ────────────────────────────────────────────────────────────
if run:
    model = load_model()
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        st.error("⚠️ No se pudo abrir la cámara. Verificá que esté conectada y no esté en uso.")
    else:
        stop_btn = st.button("⏹  Detener cámara", key="stop")

        while not stop_btn:
            success, frame = camera.read()
            if not success:
                st.warning("No se pudo leer el frame de la cámara.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            results = model.track(frame, classes=[0], persist=True, verbose=False)
            aura_mask = np.zeros_like(frame)

            if results[0].masks is not None and results[0].boxes.id is not None:
                masks    = results[0].masks.data
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for mask, track_id in zip(masks, track_ids):
                    mask_np = cv2.resize(
                        mask.cpu().numpy(), (w, h)
                    ).astype(np.uint8)

                    color_bgr = CHAKRA_BGR[track_id % 7]
                    colored = np.zeros_like(frame)
                    colored[mask_np == 1] = color_bgr
                    aura_mask = cv2.add(aura_mask, colored)

                # Expandir y suavizar el aura
                kernel_size = 21
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                aura_mask = cv2.dilate(aura_mask, kernel, iterations=aura_expand)

                blur_k = aura_blur if aura_blur % 2 == 1 else aura_blur + 1
                aura_mask = cv2.GaussianBlur(aura_mask, (blur_k, blur_k), 0)

            frame = cv2.addWeighted(frame, 1.0, aura_mask, aura_intensity, 0)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_slot.image(frame_rgb, use_container_width=True)

        camera.release()
else:
    frame_slot.markdown(
        """
        <div style="
            height:480px;
            display:flex;
            align-items:center;
            justify-content:center;
            color:#444;
            font-family:'Yatra One',cursive;
            font-size:22px;
            letter-spacing:2px;
        ">
            Activá la cámara desde el menú →
        </div>
        """,
        unsafe_allow_html=True,
    )
