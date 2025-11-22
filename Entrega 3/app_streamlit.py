"""
Sistema de Reconocimiento de Actividades Humanas en Tiempo Real
Proyecto APO III - Universidad ICESI
Autores: Angela Gonzalez, Juliana Filigrana, Juan Manuel Casanova
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
from PIL import Image
import time

# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Reconocimiento de Actividades",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-title {
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 1.1em;
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CARGAR MODELO
# =============================================================================

@st.cache_resource
def cargar_modelo():
    try:
        pipeline = joblib.load('modelo_final.pkl')
        return pipeline
    except FileNotFoundError:
        st.error("No se encontró modelo_final.pkl en la carpeta actual")
        st.stop()
    except Exception as e:
        st.error(f" Error al cargar modelo: {e}")
        st.stop()

# Cargar componentes del modelo
with st.spinner('Cargando modelo...'):
    pipeline = cargar_modelo()
    modelo = pipeline['modelo']
    imputer = pipeline['imputer']
    top_indices = pipeline['top_indices']
    label_encoder = pipeline['label_encoder']
    metadata = pipeline['metadata']

# =============================================================================
# TÍTULO Y SIDEBAR
# =============================================================================

st.markdown('<p class="main-title">🎯 Sistema de Reconocimiento de Actividades Humanas</p>', 
            unsafe_allow_html=True)
st.markdown('<p class="subtitle">Proyecto APO III - Universidad ICESI</p>', 
            unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuración")
    
    st.success(f"✅ Modelo cargado correctamente")
    st.info(f"**Clases:** {len(label_encoder.classes_)}")
    
    with st.expander("📋 Clases detectables"):
        for i, clase in enumerate(label_encoder.classes_, 1):
            st.write(f"{i}. {clase}")
    
    st.markdown("---")
    
    confidence_threshold = st.slider(
        "Umbral de confianza",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05,
        help="Predicciones con confianza menor se marcarán como inciertas"
    )
    
    st.subheader("MediaPipe Config")
    detection_conf = st.slider("Min Detection Confidence", 0.0, 1.0, 0.5, 0.05)
    tracking_conf = st.slider("Min Tracking Confidence", 0.0, 1.0, 0.5, 0.05)
    
    st.subheader("Visualización")
    show_skeleton = st.checkbox("Mostrar esqueleto", value=True)
    show_fps = st.checkbox("Mostrar FPS", value=True)
    
    st.markdown("---")
    
    with st.expander("ℹ️ Información del modelo"):
        st.write(f"**Train accuracy:** {metadata['train_accuracy']:.2%}")
        st.write(f"**Test accuracy:** {metadata['test_accuracy']:.2%}")
        st.write(f"**CV accuracy:** {metadata['cv_accuracy_mean']:.2%}")
        st.write(f"**Features:** {metadata['n_features_seleccionadas']}/{metadata['n_features_original']}")

# =============================================================================
# FUNCIONES DE PROCESAMIENTO
# =============================================================================

def extraer_landmarks(results):
    """Extrae coordenadas de landmarks de MediaPipe"""
    if not results.pose_landmarks:
        return None
    
    landmarks = []
    for landmark in results.pose_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])
    
    # MediaPipe Pose da 33 landmarks * 3 coords = 99 valores
    # Rellenar con ceros hasta 1629 (estructura del modelo original)
    while len(landmarks) < 1629:
        landmarks.append(0.0)
    
    return np.array(landmarks[:1629]).reshape(1, -1)


def calcular_angulos(landmarks):
    """Calcula ángulos articulares para visualización"""
    angulos = {}
    
    try:
        # Rodilla derecha (cadera-rodilla-tobillo)
        cadera = np.array([landmarks.landmark[24].x, landmarks.landmark[24].y])
        rodilla = np.array([landmarks.landmark[26].x, landmarks.landmark[26].y])
        tobillo = np.array([landmarks.landmark[28].x, landmarks.landmark[28].y])
        
        v1 = cadera - rodilla
        v2 = tobillo - rodilla
        angulo = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
        angulos['Rodilla Derecha'] = np.degrees(angulo)
        
        # Rodilla izquierda
        cadera = np.array([landmarks.landmark[23].x, landmarks.landmark[23].y])
        rodilla = np.array([landmarks.landmark[25].x, landmarks.landmark[25].y])
        tobillo = np.array([landmarks.landmark[27].x, landmarks.landmark[27].y])
        
        v1 = cadera - rodilla
        v2 = tobillo - rodilla
        angulo = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
        angulos['Rodilla Izquierda'] = np.degrees(angulo)
        
        # Codo derecho
        hombro = np.array([landmarks.landmark[12].x, landmarks.landmark[12].y])
        codo = np.array([landmarks.landmark[14].x, landmarks.landmark[14].y])
        muneca = np.array([landmarks.landmark[16].x, landmarks.landmark[16].y])
        
        v1 = hombro - codo
        v2 = muneca - codo
        angulo = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
        angulos['Codo Derecho'] = np.degrees(angulo)
        
    except:
        pass
    
    return angulos


def predecir_actividad(features):
    """Realiza predicción con el modelo"""
    # Imputar valores faltantes
    features_imputed = imputer.transform(features)
    
    # Seleccionar solo las features importantes
    features_reduced = features_imputed[:, top_indices]
    
    # Predecir
    pred_encoded = modelo.predict(features_reduced)[0]
    pred_proba = modelo.predict_proba(features_reduced)[0]
    
    prediccion = label_encoder.inverse_transform([pred_encoded])[0]
    confianza = pred_proba.max()
    
    # Probabilidades de todas las clases
    probas = {label_encoder.classes_[i]: pred_proba[i] for i in range(len(pred_proba))}
    
    return prediccion, confianza, probas


# =============================================================================
# INTERFAZ PRINCIPAL
# =============================================================================

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Video en Tiempo Real")
    video_placeholder = st.empty()
    
    # Botones de control
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        start_btn = st.button("▶️ Iniciar Cámara", use_container_width=True)
    with col_btn2:
        stop_btn = st.button("⏹️ Detener", use_container_width=True)
    with col_btn3:
        snapshot_btn = st.button("📸 Captura", use_container_width=True)

with col2:
    st.subheader("📊 Información en Tiempo Real")
    
    # Placeholders
    prediction_placeholder = st.empty()
    confidence_placeholder = st.empty()
    probabilities_placeholder = st.empty()
    angles_placeholder = st.empty()
    fps_placeholder = st.empty()

# =============================================================================
# PROCESAMIENTO DE VIDEO
# =============================================================================

if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

if start_btn:
    st.session_state.camera_active = True

if stop_btn:
    st.session_state.camera_active = False

if st.session_state.camera_active:
    # Inicializar MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=detection_conf,
        min_tracking_confidence=tracking_conf,
        model_complexity=1
    )
    
    # Abrir webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error(" No se pudo abrir la cámara")
        st.session_state.camera_active = False
    else:
        prev_time = time.time()
        
        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Error al leer frame")
                break
            
            # Calcular FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
            prev_time = curr_time
            
            # Procesar con MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            # Dibujar skeleton
            if show_skeleton and results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
            
            # Predecir
            if results.pose_landmarks:
                features = extraer_landmarks(results)
                
                if features is not None:
                    prediccion, confianza, probas = predecir_actividad(features)
                    angulos = calcular_angulos(results.pose_landmarks)
                    
                    # Actualizar métricas
                    with prediction_placeholder.container():
                        st.metric("🎯 Actividad Detectada", prediccion.replace('_', ' ').upper())
                    
                    with confidence_placeholder.container():
                        color = "🟢" if confianza >= confidence_threshold else "🟡"
                        st.metric(f"{color} Confianza", f"{confianza:.2%}")
                    
                    # Probabilidades
                    with probabilities_placeholder.container():
                        st.write("**Probabilidades:**")
                        for cls, prob in sorted(probas.items(), key=lambda x: x[1], reverse=True):
                            st.progress(float(prob), text=f"{cls}: {prob:.1%}")
                    
                    # Ángulos
                    if angulos:
                        with angles_placeholder.container():
                            st.write("**Ángulos Articulares:**")
                            for nombre, valor in angulos.items():
                                st.write(f"• {nombre}: **{valor:.1f}°**")
            
            # FPS
            if show_fps:
                with fps_placeholder.container():
                    st.metric("⚡ FPS", f"{fps:.1f}")
            
            # Mostrar frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Pausa pequeña
            time.sleep(0.03)
        
        cap.release()
        pose.close()
else:
    video_placeholder.info("📹 Presiona '▶️ Iniciar Cámara' para comenzar")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Proyecto APO III - Sistema de Reconocimiento de Actividades Humanas</strong></p>
    <p>Angela Gonzalez • Juliana Filigrana • Juan Manuel Casanova</p>
    <p>Universidad ICESI - 2025</p>
</div>
""", unsafe_allow_html=True)
