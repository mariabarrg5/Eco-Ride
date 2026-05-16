import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys
# Import the specific module where the attribute is expected
import sklearn.compose._column_transformer as ct

# 1. Título de la aplicación
st.set_page_config(page_title='Sistema de Alerta Temprana de Churn - Eco-Ride 🛴', layout='centered')
st.title('Sistema de Alerta Temprana de Churn - Eco-Ride 🛴')

# Cargar el modelo y el pipeline de preprocesamiento
# Asegúrate de que 'modelo_churn.pkl' y 'pipeline_preproc.pkl' estén en la raíz de tu entorno de Colab
try:
    # --- Start of temporary fix for sklearn _RemainderColsList error ---
    # This workaround addresses compatibility issues when loading ColumnTransformer
    # pipelines saved with different scikit-learn versions, specifically related
    # to the internal _RemainderColsList attribute.
    original_remainder_cols_list_patched = False
    if not hasattr(ct, '_RemainderColsList'):
        class _RemainderColsList:
            pass
        ct._RemainderColsList = _RemainderColsList
        original_remainder_cols_list_patched = True
        st.warning("Aplicando parche temporal para compatibilidad de sklearn '_RemainderColsList'.")
    # --- End of temporary fix ---

    model = joblib.load('modelo_churn.pkl')
    preprocessor = joblib.load('pipeline_preproc.pkl')
    st.success("Modelo y pipeline cargados exitosamente.")

    # --- Restore original state after loading ---
    if original_remainder_cols_list_patched:
        del ct._RemainderColsList # Remove the dummy class
        # st.info("Parche '_RemainderColsList' removido.") # Optional: for debugging
    # --- End of restore ---

except Exception as e:
    st.error(f"Error al cargar el modelo o el pipeline: {e}. Asegúrate de que 'modelo_churn.pkl' y 'pipeline_preproc.pkl' estén en el directorio de trabajo actual (normalmente /content/ en Colab).")
    st.stop() # Detener la ejecución si no se pueden cargar los archivos

st.sidebar.header('Datos del Cliente')

# 2. Panel lateral (sidebar) con controles para ingresar datos de un cliente
edad = st.sidebar.slider('Edad', 18, 80, 30)
plan = st.sidebar.selectbox('Plan', ('básico', 'premium', 'elite'))
uso_mensual_km = st.sidebar.slider('Uso_Mensual_Km', 0.0, 1000.0, 150.0)
soporte_tickets = st.sidebar.slider('Soporte_Tickets', 0, 20, 1)
gasto_promedio = st.sidebar.slider('Gasto_Promedio', 0.0, 500.0, 50.0)
region = st.sidebar.selectbox('Region', ('Norte', 'Sur', 'Centro'))
dias_antiguedad = st.sidebar.slider('Dias_Antiguedad', 0, 1825, 365) # 1825 days = 5 years

# 3. Lógica interna: Construir DataFrame, aplicar pipeline y predecir

# Crear un DataFrame con los datos de entrada
# Es CRÍTICO que el orden y nombre de las columnas coincidan con X_train original.
# X_train original columns: 'Edad', 'Plan', 'Uso_Mensual_Km', 'Soporte_Tickets', 'Gasto_Promedio', 'Region', 'Dias_Antiguedad'
input_data = pd.DataFrame({
    'Edad': [edad],
    'Plan': [plan],
    'Uso_Mensual_Km': [uso_mensual_km],
    'Soporte_Tickets': [soporte_tickets],
    'Gasto_Promedio': [gasto_promedio],
    'Region': [region],
    'Dias_Antiguedad': [dias_antiguedad]
})

st.subheader("Datos ingresados del cliente:")
st.write(input_data)

# Botón para analizar riesgo de Churn
if st.button('Analizar Riesgo de Churn'):
    # Aplicar SOLO .transform() del pipeline cargado
    # El pipeline manejará la imputación, escalado y codificación
    processed_input = preprocessor.transform(input_data)

    # Predecir con el modelo cargado
    prediction = model.predict(processed_input)
    # Obtener probabilidades (si el modelo lo soporta, para clasificador binario la clase 1)
    prediction_proba = model.predict_proba(processed_input)[:, 1]

    st.subheader("Resultado del Análisis:")

    if prediction[0] == 1:
        st.markdown(f"## <span style='color:red;'>🚨 Alto Riesgo de Cancelación</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"## <span style='color:green;'>✅ Cliente Estable</span>", unsafe_allow_html=True)

    st.write(f"Probabilidad de Churn: **{prediction_proba[0]*100:.2f}%**")
