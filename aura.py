import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Configuración de página limpia
st.set_page_config(page_title="Aura", layout="wide", initial_sidebar_state="expanded")

# 2. Inyección de CSS (Fondo blanco, centrado absoluto y ocultar elementos sobrantes)
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Yatra+One&display=swap');
        
        /* Forzar fondo blanco en toda la aplicación */
        .stApp {
            background-color: #FFFFFF !important;
        }
        
        /* Ocultar elementos nativos de Streamlit */
        #MainMenu, footer, header { visibility: hidden; }
        
        /* Menú lateral gris oscuro/negro para que resalte la palabra Aura */
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

        /* Contenedor central del video estilo monitor flotante */
        .video-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 0 auto;
            max-width: 800px; /* Tamaño del cuadro en modo normal */
            border-radius: 8px;
            box-shadow: 0px 10px 30px rgba(0,0,0,0.15);
            overflow: hidden;
            transition: all 0.3s ease;
        }

        /* Clase especial que se activará dinámicamente al presionar F11 */
        body.fullscreen-mode .video-container {
            max-width: 100vw !important;
            width: 100vw !important;
            height: 100vh !important;
            position: fixed;
            top: 0;
            left: 0;
            z-index: 99999;
            border-radius: 0px;
        }
        body.fullscreen-mode [data-testid="stSidebar"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# 3. Script de JavaScript para escuchar el cambio de pantalla completa (F11 o nativo)
st.markdown("""
    <script>
        const doc = window.parent.document;
        doc.addEventListener('fullscreenchange', () => {
            if (doc.fullscreenElement) {
                // Si está en pantalla completa, agrega la clase mística para agrandar el video
                doc.body.classList.add('fullscreen-mode');
            } else {
                // Si sale, vuelve al diseño centrado con fondo blanco
                doc.body.classList.remove('fullscreen-mode');
            }
        });
    </script>
""", unsafe_allow_html=True)

# 4. Menú lateral con fuente de la India
with st.sidebar:
    st.markdown('<p class="titulo-india">Aura</p>', unsafe_allow_html=True)

# 5. Envolver el área de video en nuestro contenedor CSS personalizado
st.markdown('<div class="video-container">', unsafe_allow_html=True)
FRAME_WINDOW = st.image([])
st.markdown('</div>', unsafe_allow_html=True)

# Cargar modelo YOLO26
@st.cache_resource
def load_model():
    return YOLO("yolo26n-seg.pt")

model = load_model()

# Colores de los Chakras (RGB)
CHAKRA_COLORS = {
    0: [255, 0, 0], 1: [255, 127, 0], 2: [255, 255, 0],
    3: [0, 255, 0], 4: [0, 191, 255], 5: [75, 0, 130], 6: [148, 0, 211]
}

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    results = model.track(frame, classes=[0], persist=True, verbose=False)
    aura_mask = np.zeros_like(frame)

    if results[0].masks is not None and results[0].boxes.id is not None:
        masks = results[0].masks.data
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for mask, track_id in zip(masks, track_ids):
            mask = mask.cpu().numpy()
            mask_resized = cv2.resize(mask, (w, h)).astype(np.uint8)
            
            index_chakra = track_id % 7
            color_rgb = CHAKRA_COLORS[index_chakra]
            aura_mask[mask_resized == 1] = [color_rgb[2], color_rgb[1], color_rgb[0]]

        kernel = np.ones((21, 21), np.uint8)
        aura_mask = cv2.dilate(aura_mask, kernel, iterations=2)
        aura_mask = cv2.GaussianBlur(aura_mask, (65, 65), 0)
        frame = cv2.addWeighted(frame, 1.0, aura_mask, 0.55, 0)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Renderizar frame continuo
    FRAME_WINDOW.image(frame_rgb, use_container_width=True)

camera.release()