import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageOps
import os

# ------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA
# ------------------------------------------------------
st.set_page_config(
    page_title="Guardián Forestal (Local)",
    layout="wide",
    page_icon="🌲"
)

# ------------------------------------------------------
# 2. CARGA INTELIGENTE DE MODELOS
# ------------------------------------------------------
@st.cache_resource
def cargar_modelos():
    modelos_cargados = {}
    
    # 1. Intentar cargar el Multimodal
    if os.path.exists('multimodal_baseline.keras'):
        try:
            modelos_cargados['Multimodal (Datos + Satélite)'] = tf.keras.models.load_model('multimodal_baseline.keras')
        except Exception as e:
            print(f"Error cargando Multimodal: {e}")

    # 2. Intentar cargar la CNN (Solo Imágenes)
    # AJUSTA AQUÍ EL NOMBRE DE TU ARCHIVO CNN SI ES DISTINTO
    nombres_posibles_cnn = ['cnn_image_baseline.keras', 'modelo_cnn.keras', 'cnn_model.keras']
    
    for nombre in nombres_posibles_cnn:
        if os.path.exists(nombre):
            try:
                modelos_cargados['CNN Visual (Solo Satélite)'] = tf.keras.models.load_model(nombre)
                break # Si encuentra uno, deja de buscar
            except:
                continue

    return modelos_cargados

dict_modelos = cargar_modelos()

if not dict_modelos:
    st.error("❌ No se encontraron modelos .keras en la carpeta. Por favor, revisa que los archivos estén junto al app.py")
    st.stop()

# ------------------------------------------------------
# 3. BARRA LATERAL (SELECTOR Y DATOS)
# ------------------------------------------------------
st.sidebar.header("🎛️ Centro de Control")

# Selector de Modelo
nombre_modelo = st.sidebar.selectbox("Selecciona el Cerebro IA:", list(dict_modelos.keys()))
modelo_activo = dict_modelos[nombre_modelo]

# Lógica para mostrar/ocultar inputs según el modelo
datos_tabulares = None

if "Multimodal" in nombre_modelo:
    st.sidebar.subheader("📊 Datos del Terreno")
    temp = st.sidebar.slider("Temperatura (ºC)", -10, 50, 25)
    humedad = st.sidebar.slider("Humedad (%)", 0, 100, 40)
    viento = st.sidebar.slider("Viento (km/h)", 0, 150, 15)
    lluvia = st.sidebar.slider("Lluvia (mm)", 0, 200, 0)
    
    # Preparamos el vector para el Multimodal
    # OJO: El orden debe ser igual al entrenamiento. 
    # Aquí asumo: [Lat, Lon, Temp, Lluvia, Season] como hicimos antes
    # Si tu modelo pide otra cosa, ajústalo aquí.
    lat_dummy = 0.5  # Valor medio normalizado
    lon_dummy = 0.5  # Valor medio normalizado
    season_dummy = 0.25
    
    # Normalización manual simple (dividir por maximos aproximados) para que no explote
    datos_tabulares = np.array([[
        lat_dummy, 
        lon_dummy, 
        temp / 50.0, 
        lluvia / 50.0, 
        season_dummy
    ]], dtype=np.float32)
    
else:
    st.sidebar.info("ℹ️ El modelo CNN solo necesita la imagen. Los datos numéricos están desactivados.")

# ------------------------------------------------------
# 4. ZONA PRINCIPAL
# ------------------------------------------------------
st.title("🛡️ Guardián Forestal: Análisis de Riesgo")
st.markdown(f"**Modelo Activo:** `{nombre_modelo}`")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 1. Sube la Imagen")
    archivo = st.file_uploader("Vista Satelital o Cámara", type=["jpg", "png", "jpeg"])
    
    img_tensor = None
    if archivo:
        image = Image.open(archivo).convert('RGB')
        st.image(image, use_container_width=True)
        
        # Preprocesamiento de imagen
        # Usamos 128x128 para Multimodal y 224x224 para CNN (ajusta si es distinto)
        target_size = (128, 128) if "Multimodal" in nombre_modelo else (224, 224)
        
        img_resized = image.resize(target_size)
        img_array = np.array(img_resized) / 255.0 # Normalizar a 0-1
        img_tensor = np.expand_dims(img_array, axis=0) # Batch de 1

with col2:
    st.markdown("### 2. Resultado del Análisis")
    
    if st.button("🔥 PREDECIR RIESGO", use_container_width=True):
        if img_tensor is None:
            st.warning("⚠️ Sube una imagen primero.")
        else:
            try:
                with st.spinner("Analizando patrones..."):
                    # Lógica de Predicción Diferenciada
                    if "Multimodal" in nombre_modelo:
                        # Entran DOS cosas: [Imagen, Datos] (o al revés, según tu entreno)
                        pred = modelo_activo.predict([img_tensor, datos_tabulares])
                    else:
                        # Entra SOLO imagen
                        pred = modelo_activo.predict(img_tensor)
                
                # Procesar resultado
                probabilidad = float(pred[0][0]) * 100
                
                # Visualización Gauge (Velocímetro)
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probabilidad,
                    title = {'text': "Probabilidad de Incendio"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if probabilidad > 50 else "green"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgreen"},
                            {'range': [40, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "salmon"}
                        ]
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # Mensaje final
                if probabilidad > 60:
                    st.error("🚨 ¡ALERTA MÁXIMA! Se detectan condiciones de incendio.")
                elif probabilidad > 30:
                    st.warning("⚠️ Riesgo Moderado. Mantener vigilancia.")
                else:
                    st.success("✅ Zona Segura. Bajo riesgo detectado.")
                    
            except Exception as e:
                st.error(f"Algo falló en el cálculo: {e}")
                st.write("Consejo: Si usas Multimodal, revisa que el orden de inputs sea [imagen, datos] o [datos, imagen].")

# Pie de página
st.markdown("---")
st.caption("Ejecutando en entorno LOCAL - Sin límites de nube 🚀")
