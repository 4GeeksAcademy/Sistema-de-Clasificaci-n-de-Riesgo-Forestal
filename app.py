import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageOps
import os
import datetime
import pydeck as pdk
import joblib
import time 
import random 
import keras 
import sklearn

# --- 1. CONFIGURACIÓN ---
st.set_page_config(
    page_title="Guardián Forestal",
    layout="wide",
    page_icon="🔥",
    initial_sidebar_state="collapsed"
)

# --- 2. FUNCIONES DE CARGA ---
@st.cache_resource
def cargar_recursos():
    # Intenta cargar encoders, si fallan no pasa nada, seguimos sin ellos
    le_state = None
    le_cause = None
    try:
        if os.path.exists('encoder_estados.pkl'):
            le_state = joblib.load('encoder_estados.pkl')
        if os.path.exists('encoder_causas.pkl'):
            le_cause = joblib.load('encoder_causas.pkl')
    except:
        pass # Ignoramos error de encoders por ahora
    
    modelos = {}
    # Carga de modelos .keras
    if os.path.exists('multimodal_baseline.keras'):
        try: modelos['Multimodal (Datos + Foto)'] = keras.models.load_model('multimodal_baseline.keras')
        except: pass
    
    # Si tienes otro modelo CNN, descomenta esto:
    # if os.path.exists('cnn_image_baseline.keras'):
    #     try: modelos['CNN Visual (Solo Foto)'] = keras.models.load_model('cnn_image_baseline.keras')
    #     except: pass

    return modelos, le_state, le_cause

try:
    dict_modelos, le_state, le_cause = cargar_recursos()
    if not dict_modelos:
        st.error("⚠️ No se encontraron modelos .keras en la carpeta. Sube 'multimodal_baseline.keras'.")
        st.stop()
except Exception as e: 
    st.error(f"Error cargando recursos: {e}")
    st.stop()

# --- 3. ESTILOS CSS ---
st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    .clean-card { background-color: #FFFFFF; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); padding: 20px; border: 1px solid #E9ECEF; }
    .alert-card { background-color: #FFF5F5 !important; border: 2px solid #DC3545 !important; animation: pulse-red 2s infinite; }
    @keyframes pulse-red { 0% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.7); } 70% { box-shadow: 0 0 0 15px rgba(220, 53, 69, 0); } 100% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0); } }
    .blink-text { color: #DC3545; font-weight: 900; animation: blink 1s infinite; text-align: center; font-size: 1.2em; }
</style>
""", unsafe_allow_html=True)

# --- 4. MEMORIA DE VARIABLES ---
if 'temp_sim' not in st.session_state: st.session_state.temp_sim = 25
if 'rain_sim' not in st.session_state: st.session_state.rain_sim = 10
if 'lat_sim' not in st.session_state: st.session_state.lat_sim = 36.77
if 'lon_sim' not in st.session_state: st.session_state.lon_sim = -119.41

# --- 5. ENCABEZADO ---
st.markdown("# 🌲 Guardián Forestal IA")
c1, c2 = st.columns([3, 1])
with c1: st.markdown("### Simulador de Riesgo Forestal")
with c2: 
    nombre_modelo_sel = st.selectbox("🧠 Modelo:", list(dict_modelos.keys()))
    modelo_activo = dict_modelos[nombre_modelo_sel]
st.markdown("---")

# Variables por defecto
datos_para_modelo = {'TMAX_C': 0, 'PRCP': 0, 'LATITUDE': 0, 'LONGITUDE': 0, 'STATE': "CA"}
imagen_para_procesar = None 

# --- 6. INTERFAZ Y INPUTS ---
st.markdown("#### 📸 1. Cargar Imagen Satelital")
uploaded_file = st.file_uploader("Sube imagen (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    imagen_para_procesar = Image.open(uploaded_file).convert('RGB')
    
    # Botón mágico para simular datos
    if st.button("🪄 Auto-Detectar Clima (Simulación)"):
        st.session_state.temp_sim = random.randint(35, 45)
        st.session_state.rain_sim = 0
        st.session_state.lat_sim = 34.05 + random.uniform(-0.1, 0.1) 
        st.session_state.lon_sim = -118.24 + random.uniform(-0.1, 0.1)
        st.rerun()

st.markdown("#### 🎛️ 2. Datos del Entorno")
with st.expander("Panel de Control Manual", expanded=True):
    cols = st.columns(3)
    temp = cols[0].slider("🌡️ Temp (ºC)", -10, 50, key='temp_sim')
    rain = cols[0].slider("💧 Lluvia (mm)", 0, 100, key='rain_sim')
    lat = cols[2].number_input("Latitud", key='lat_sim')
    lon = cols[2].number_input("Longitud", key='lon_sim')
    # Valor fijo para evitar errores con encoders por ahora
    datos_para_modelo.update({'TMAX_C': temp, 'PRCP': rain, 'LATITUDE': lat, 'LONGITUDE': lon})

# --- 7. PREPARACIÓN DE DATOS (CRÍTICO) ---

# A) PREPARAR DATOS NUMÉRICOS (TABULAR)
# El orden debe ser EXACTAMENTE el mismo que usaste al entrenar
# Asumo: Latitud, Longitud, Temperatura, Lluvia, Season(fijo 1)
season_fijo = 1.0 
vector_numerico = np.array([[
    datos_para_modelo['LATITUDE'], 
    datos_para_modelo['LONGITUDE'], 
    datos_para_modelo['TMAX_C'], 
    datos_para_modelo['PRCP'], 
    season_fijo
]], dtype=np.float32)

# B) PREPARAR IMAGEN
target_size = (128, 128) # Tamaño estándar para multimodal
img_tensor = None

if imagen_para_procesar:
    # Redimensionar y normalizar
    img_resized = imagen_para_procesar.resize(target_size)
    img_array = np.array(img_resized) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0) # Crear batch de 1

# --- 8. LÓGICA DE PREDICCIÓN ---
riesgo = 0.0
prediccion_realizada = False

if st.button("🔥 CALCULAR RIESGO"):
    if img_tensor is None:
        st.warning("⚠️ Por favor, sube una imagen primero.")
    else:
        # ==========================================
        # 🕵️‍♂️ MODO DETECTIVE (DEBUG)
        # ==========================================
        st.info("🕵️‍♂️ **INSPECTOR DE DATOS ACTIVADO**")
        c_debug1, c_debug2 = st.columns(2)
        
        with c_debug1:
            st.write("📊 **Datos Numéricos (Input):**")
            st.write(vector_numerico) # Muestra el array real
            
            if np.isnan(vector_numerico).any():
                st.error("🚨 **ALERTA:** ¡Hay valores NaN (vacíos)!")
            else:
                st.success("✅ Datos numéricos limpios.")

        with c_debug2:
            st.write(f"🖼️ **Forma Imagen:** {img_tensor.shape}")
            st.write(f"📦 **Scikit-learn Versión:** {sklearn.__version__}")
        # ==========================================

        try:
            # PREDICCIÓN REAL
            if "Multimodal" in nombre_modelo_sel:
                # El modelo espera una lista: [imagen, datos]
                pred = modelo_activo.predict([img_tensor, vector_numerico])
            else:
                # Solo imagen
                pred = modelo_activo.predict(img_tensor)
            
            riesgo = float(pred[0][0]) * 100
            prediccion_realizada = True

        except Exception as e:
            st.error(f"💥 Error fatal en la predicción: {e}")
            st.write("Consejo: Revisa si tu modelo espera [datos, imagen] o [imagen, datos].")

# --- 9. RESULTADOS VISUALES ---

if prediccion_realizada:
    st.write("---")
    cl, cr = st.columns([1, 1])

    # IZQUIERDA: IMAGEN
    with cl:
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.subheader("👁️ Vista Satelital")
        st.image(imagen_para_procesar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # DERECHA: MEDIDOR DE RIESGO
    with cr:
        color = "#DC3545" if riesgo > 50 else ("#28a745" if riesgo < 40 else "#ffc107")
        card_class = "clean-card alert-card" if riesgo > 50 else "clean-card"
        
        st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align:center; color:{color}'>RIESGO CALCULADO: {riesgo:.1f}%</h2>", unsafe_allow_html=True)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = riesgo,
            gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': color}}
        ))
        fig.update_layout(height=200, margin=dict(l=20,r=20,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        if riesgo > 50:
            st.markdown(f'<div class="blink-text">🚨 ¡PELIGRO DE INCENDIO! 🚨</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)