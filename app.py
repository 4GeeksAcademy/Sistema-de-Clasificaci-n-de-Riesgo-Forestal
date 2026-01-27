import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import os
import datetime
import pydeck as pdk
import joblib
import time 
import random 
import keras 

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
    le_state = joblib.load('encoder_estados.pkl') if os.path.exists('encoder_estados.pkl') else None
    le_cause = joblib.load('encoder_causas.pkl') if os.path.exists('encoder_causas.pkl') else None
    
    modelos = {}
    if os.path.exists('multimodal_baseline.keras'):
        try: modelos['Multimodal (Datos + Foto)'] = keras.models.load_model('multimodal_baseline.keras')
        except: pass
    if os.path.exists('cnn_image_baseline.keras'):
        try: modelos['CNN Visual (Solo Foto)'] = keras.models.load_model('cnn_image_baseline.keras')
        except: pass

    if not modelos: st.error("❌ ERROR: Faltan modelos .keras"); st.stop()
    return modelos, le_state, le_cause

try:
    dict_modelos, le_state, le_cause = cargar_recursos()
except Exception as e: st.error(f"Error: {e}"); st.stop()

# --- 3. ESTILOS ---
st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; color: #212529; font-family: 'Helvetica Neue', sans-serif; }
    .clean-card { background-color: #FFFFFF; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); padding: 20px; border: 1px solid #E9ECEF; }
    .alert-card { background-color: #FFF5F5 !important; border: 2px solid #DC3545 !important; animation: pulse-red 2s infinite; }
    @keyframes pulse-red { 0% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.7); } 70% { box-shadow: 0 0 0 15px rgba(220, 53, 69, 0); } 100% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0); } }
    .blink-text { color: #DC3545; font-weight: 900; animation: blink 1s infinite; text-align: center; font-size: 1.2em; }
    div.stButton > button:first-child { background-color: #6f42c1; color: white; border-radius: 8px; border: none; font-weight: bold; width: 100%; }
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Ajuste para que el mapa se vea bonito dentro de la tarjeta */
    .map-container { border: 2px solid #DC3545; border-radius: 8px; overflow: hidden; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 4. MEMORIA ---
if 'temp_sim' not in st.session_state: st.session_state.temp_sim = 25
if 'rain_sim' not in st.session_state: st.session_state.rain_sim = 10
if 'lat_sim' not in st.session_state: st.session_state.lat_sim = 36.77
if 'lon_sim' not in st.session_state: st.session_state.lon_sim = -119.41

# --- 5. UI HEADER ---
st.markdown("# 🌲 Guardián Forestal IA")
c1, c2 = st.columns([3, 1])
with c1: st.markdown("### Simulador de Riesgo Forestal")
with c2: 
    nombre_modelo_sel = st.selectbox("🧠 Modelo:", list(dict_modelos.keys()))
    modelo_activo = dict_modelos[nombre_modelo_sel]
st.markdown("---")

# Variables por defecto
datos_para_modelo = {'TMAX_C': 0, 'PRCP': 0, 'LATITUDE': 0, 'LONGITUDE': 0, 'STATE': "CA", 'NWCG_GENERAL_CAUSE': "Unknown", 'SEASON': 1}
imagen_para_procesar = None 

# --- 6. INTERFAZ PRINCIPAL ---
    
st.markdown("#### 📸 Cargar Imagen Satelital")
uploaded_file = st.file_uploader("Sube imagen (JPG, PNG, TIFF)", type=["jpg", "png", "tiff"])

if uploaded_file:
    imagen_para_procesar = Image.open(uploaded_file).convert('RGB')
    
    if "Multimodal" in nombre_modelo_sel:
        if st.button("🪄 Auto-Detectar Clima (Simulación API)"):
            with st.spinner("🛰️ Conectando con sensores..."): time.sleep(1)
            st.session_state.temp_sim = random.randint(35, 45)
            st.session_state.rain_sim = 0
            # Coordenadas aleatorias de California para que el mapa cambie
            st.session_state.lat_sim = 34.05 + random.uniform(-0.5, 0.5) 
            st.session_state.lon_sim = -118.24 + random.uniform(-0.5, 0.5)
            st.rerun()

# === PANELES DE CONTROL ===
if "Multimodal" in nombre_modelo_sel:
    st.markdown("#### 🎛️ Datos del Entorno")
    with st.expander("Panel de Control Manual", expanded=True):
        cols = st.columns(3)
        temp = cols[0].slider("🌡️ Temp", -10, 50, key='temp_sim')
        rain = cols[0].slider("💧 Lluvia", 0, 100, key='rain_sim')
        lat = cols[2].number_input("Latitud", key='lat_sim')
        lon = cols[2].number_input("Longitud", key='lon_sim')
        estado = cols[1].selectbox("Estado", list(le_state.classes_) if le_state else ["CA"])
        
        datos_para_modelo.update({'TMAX_C': temp, 'PRCP': rain, 'LATITUDE': lat, 'LONGITUDE': lon, 'STATE': estado})
else:
    st.info("ℹ️ **Modo Visual Puro:** Este modelo analiza solo los píxeles de la imagen.")
    # Usamos las coordenadas de memoria para que el mapa no salga en 0,0
    datos_para_modelo.update({'LATITUDE': st.session_state.lat_sim, 'LONGITUDE': st.session_state.lon_sim})

# --- 7. PROCESAMIENTO ---
vector_numerico = np.array([[
    datos_para_modelo['LATITUDE'], datos_para_modelo['LONGITUDE'], 
    datos_para_modelo['TMAX_C'], datos_para_modelo['PRCP'], 
    1 
]], dtype=np.float32)

target_size = (128, 128) if "Multimodal" in nombre_modelo_sel else (224, 224)

if imagen_para_procesar:
    img_tensor = np.expand_dims(np.array(imagen_para_procesar.resize(target_size)) / 255.0, axis=0)
else:
    img_tensor = np.zeros((1, target_size[0], target_size[1], 3), dtype=np.float32)

# --- 8. PREDICCIÓN ---
# ==========================================
# 🛑 ZONA DE INSPECCIÓN (DEBUG)
# ==========================================
st.write("--- 🕵️‍♂️ INSPECTOR DE DATOS ---")

# 1. Muestra qué datos numéricos están entrando al modelo
st.write("📊 Datos Tabulares Crudos:", tabular_data)

# 2. Comprueba si hay NaNs (Agujeros negros en los datos)
import numpy as np
try:
    hay_nans = np.isnan(tabular_data).any()
    if hay_nans:
        st.error("🚨 ALERTA: ¡Hay valores NaN (vacíos) entrando al modelo!")
    else:
        st.success("✅ Los datos numéricos parecen sanos (No hay NaNs).")
except Exception as e:
    st.error(f"Error al comprobar NaNs: {e}")

# 3. Muestra la versión de scikit-learn que está usando Render
import sklearn
st.write(f"📦 Versión de Scikit-learn en Render: {sklearn.__version__}")
# ==========================================



# --- 9. VISUALIZACIÓN ---
st.write("")
cl, cr = st.columns([1, 1])

# COLUMNA IZQUIERDA: IMAGEN + DATOS CLIMA
with cl:
    if imagen_para_procesar: 
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.subheader("Imagen Analizada")
        st.image(imagen_para_procesar, use_container_width=True)
        st.caption(f"Resolución de entrada: {target_size}")
        st.markdown('</div>', unsafe_allow_html=True)
    elif "CNN" in nombre_modelo_sel:
        st.warning("⚠️ Sube una foto para ver predicción.")
    else:
        st.info("Esperando imagen...")
        
    # Tarjeta extra de clima solo si es Multimodal
    if "Multimodal" in nombre_modelo_sel:
        st.write("")
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Temperatura", f"{datos_para_modelo['TMAX_C']} °C")
        m2.metric("Lluvia", f"{datos_para_modelo['PRCP']} mm")
        st.markdown('</div>', unsafe_allow_html=True)

# COLUMNA DERECHA: ALERTA + MAPA TÁCTICO
with cr:
    color = "#DC3545" if riesgo > 50 else ("#28a745" if riesgo < 40 else "#ffc107")
    card_class = "clean-card alert-card" if riesgo > 50 else "clean-card"
    
    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align:center; color:{color}'>RIESGO: {riesgo:.1f}%</h2>", unsafe_allow_html=True)
    fig = go.Figure(go.Indicator(mode = "gauge", value = riesgo, gauge = {'axis': {'visible': False}, 'bar': {'color': color}, 'bgcolor': "#E9ECEF"}))
    fig.update_layout(height=180, margin=dict(l=20,r=20,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    
    if riesgo > 50:
        st.markdown(f'<div class="blink-text">🚨 ALERTA DE INCENDIO DETECTADA</div>', unsafe_allow_html=True)
        
        # TEXTO DE COORDENADAS
        st.markdown(f"""
            <div style='text-align: center; margin-top: 10px; margin-bottom: 5px; color: #495057;'>
                <strong>📍 UBICACIÓN DEL FOCO:</strong><br>
                Lat: {datos_para_modelo['LATITUDE']:.4f} | Lon: {datos_para_modelo['LONGITUDE']:.4f}
            </div>
        """, unsafe_allow_html=True)

        # === MAPA TÁCTICO DEBAJO DEL TEXTO ===
        # Creamos un mapa centrado en el incendio
        view_state = pdk.ViewState(
            latitude=datos_para_modelo['LATITUDE'], 
            longitude=datos_para_modelo['LONGITUDE'], 
            zoom=12,  # Zoom cercano
            pitch=45  # Inclinación 3D
        )
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame({'lat': [datos_para_modelo['LATITUDE']], 'lon': [datos_para_modelo['LONGITUDE']]}),
            get_position='[lon, lat]',
            get_color='[220, 53, 69, 200]', # ROJO INTENSO
            get_radius=500, # Radio del punto
            pickable=True,
            stroked=True,
            filled=True,
            line_width_min_pixels=2
        )
        
        # Renderizamos el mapa
        st.pydeck_chart(pdk.Deck(
            map_style=pdk.map_styles.CARTO_LIGHT, 
            initial_view_state=view_state, 
            layers=[layer],
            height=250 # Altura controlada
        ))
        
    st.markdown('</div>', unsafe_allow_html=True)