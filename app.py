import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageOps
import os

# ------------------------------------------------------
# 1. CONFIGURACIÓN VISUAL (CSS + PAGE CONFIG)
# ------------------------------------------------------
st.set_page_config(
    page_title="Guardián Forestal AI",
    layout="wide",
    page_icon="🌲",
    initial_sidebar_state="expanded"
)

# ESTILOS CSS PERSONALIZADOS (La Magia Visual ✨)
st.markdown("""
<style>
    /* Fondo general más limpio */
    .stApp {
        background-color: #F4F6F9;
    }
    
    /* Estilo de Tarjetas (Cards) */
    .css-card {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #E0E0E0;
    }
    
    /* Títulos destacados */
    .highlight-title {
        color: #2C3E50;
        font-family: 'Helvetica', sans-serif;
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    /* Botón de predicción personalizado */
    .stButton > button {
        width: 100%;
        background-color: #27ae60;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 15px;
        border: none;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #219150;
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(39, 174, 96, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# 2. LÓGICA DE CARGA (Funcionalidad Intacta)
# ------------------------------------------------------
@st.cache_resource
def cargar_modelos():
    modelos_cargados = {}
    
    # Cargar Multimodal
    if os.path.exists('multimodal_baseline.keras'):
        try:
            modelos_cargados['🌲 Multimodal (Datos + Satélite)'] = tf.keras.models.load_model('multimodal_baseline.keras')
        except: pass

    # Cargar CNN
    nombres_posibles_cnn = ['cnn_image_baseline.keras', 'modelo_cnn.keras', 'cnn_model.keras']
    for nombre in nombres_posibles_cnn:
        if os.path.exists(nombre):
            try:
                modelos_cargados['📷 CNN Visual (Solo Imagen)'] = tf.keras.models.load_model(nombre)
                break
            except: continue

    return modelos_cargados

dict_modelos = cargar_modelos()

# ------------------------------------------------------
# 3. BARRA LATERAL (CONTROL PANEL)
# ------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3199/3199920.png", width=60)
    st.title("Centro de Control")
    st.markdown("---")
    
    if not dict_modelos:
        st.error("❌ No se encontraron modelos .keras")
        st.stop()

    nombre_modelo = st.selectbox("🧠 Seleccionar IA", list(dict_modelos.keys()))
    modelo_activo = dict_modelos[nombre_modelo]
    
    st.markdown("### 🎛️ Parámetros")
    
    datos_tabulares = None
    
    if "Multimodal" in nombre_modelo:
        # Inputs con diseño más limpio
        temp = st.slider("🌡️ Temperatura (ºC)", -10, 50, 25)
        col_s1, col_s2 = st.columns(2)
        humedad = col_s1.slider("💧 Humedad", 0, 100, 40)
        lluvia = col_s2.slider("🌧️ Lluvia", 0, 200, 0)
        viento = st.slider("💨 Viento (km/h)", 0, 150, 15)
        
        # Preparación de datos (Igual que antes)
        lat_dummy, lon_dummy, season_dummy = 0.5, 0.5, 0.25
        datos_tabulares = np.array([[lat_dummy, lon_dummy, temp/50.0, lluvia/50.0, season_dummy]], dtype=np.float32)
        
        st.success("✅ Datos sincronizados")
    else:
        st.info("ℹ️ Modo Visual: Solo se requiere imagen.")

# ------------------------------------------------------
# 4. ZONA PRINCIPAL (LAYOUT)
# ------------------------------------------------------

# Encabezado Principal
st.markdown('<div class="highlight-title" style="font-size: 40px;">🛡️ Guardián Forestal AI</div>', unsafe_allow_html=True)
st.markdown("Sistema de Detección Temprana de Incendios mediante Deep Learning.")
st.write("")

# Columnas principales con diseño de "Dashboard"
col_izq, col_der = st.columns([1, 1.2], gap="large")

# --- COLUMNA IZQUIERDA: INPUT VISUAL ---
with col_izq:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown('<div class="highlight-title">1. Fuente Visual</div>', unsafe_allow_html=True)
    
    archivo = st.file_uploader("Arrastra tu imagen satelital aquí", type=["jpg", "png", "jpeg"])
    
    img_tensor = None
    if archivo:
        image = Image.open(archivo).convert('RGB')
        # Bordes redondeados a la imagen
        st.image(image, use_container_width=True)
        
        # Preprocesamiento
        target_size = (128, 128) if "Multimodal" in nombre_modelo else (224, 224)
        img_resized = image.resize(target_size)
        img_array = np.array(img_resized) / 255.0
        img_tensor = np.expand_dims(img_array, axis=0)
    else:
        # Espacio vacío bonito
        st.markdown("""
        <div style="text-align:center; padding: 40px; color: #aaa; border: 2px dashed #ddd; border-radius: 10px;">
            Waiting for satellite feed... 📡
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True) # Cierre Card

# --- COLUMNA DERECHA: RESULTADOS ---
with col_der:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown('<div class="highlight-title">2. Análisis de Riesgo</div>', unsafe_allow_html=True)
    
    if st.button("🔍 ANALIZAR ZONA"):
        if img_tensor is None:
            st.warning("⚠️ Esperando señal visual (Sube una imagen).")
        else:
            try:
                with st.spinner("Procesando redes neuronales..."):
                    # Lógica exacta anterior
                    if "Multimodal" in nombre_modelo:
                        pred = modelo_activo.predict([img_tensor, datos_tabulares])
                    else:
                        pred = modelo_activo.predict(img_tensor)
                
                probabilidad = float(pred[0][0]) * 100
                
                # GRÁFICO GAUGE PROFESIONAL
                color_riesgo = "#2ecc71" # Verde
                if probabilidad > 40: color_riesgo = "#f1c40f" # Amarillo
                if probabilidad > 70: color_riesgo = "#e74c3c" # Rojo

                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probabilidad,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Probabilidad de Fuego", 'font': {'size': 24}},
                    number = {'suffix': "%", 'font': {'color': color_riesgo}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': color_riesgo},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 40], 'color': 'rgba(46, 204, 113, 0.3)'},
                            {'range': [40, 70], 'color': 'rgba(241, 196, 15, 0.3)'},
                            {'range': [70, 100], 'color': 'rgba(231, 76, 60, 0.3)'}
                        ],
                    }
                ))
                fig.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=20))
                st.plotly_chart(fig, use_container_width=True)

                # MENSAJE DE ESTADO
                if probabilidad > 70:
                    st.markdown("""
                    <div style="background-color: #ffebee; border-left: 5px solid #e53935; padding: 15px; border-radius: 5px;">
                        <h3 style="color: #c62828; margin:0;">🚨 ALERTA CRÍTICA</h3>
                        <p style="margin:0;">Condiciones extremas detectadas. Se recomienda evacuación preventiva.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif probabilidad > 40:
                    st.markdown("""
                    <div style="background-color: #fff3e0; border-left: 5px solid #fb8c00; padding: 15px; border-radius: 5px;">
                        <h3 style="color: #ef6c00; margin:0;">⚠️ RIESGO MODERADO</h3>
                        <p style="margin:0;">Monitoreo constante requerido. Condiciones favorables para ignición.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background-color: #e8f5e9; border-left: 5px solid #43a047; padding: 15px; border-radius: 5px;">
                        <h3 style="color: #2e7d32; margin:0;">✅ ZONA SEGURA</h3>
                        <p style="margin:0;">No se detectan amenazas inminentes.</p>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error interno: {e}")

    else:
        st.info("Listo para iniciar análisis.")
        
    st.markdown('</div>', unsafe_allow_html=True) 
