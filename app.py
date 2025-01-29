import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Definir las columnas categóricas y numéricas (de acuerdo con la preparación de datos)
num_cols = ['trestbps', 'chol', 'thalch', 'oldpeak', 'ca', 'age']
cat_cols = ['sex', 'cp', 'thal', 'slope', 'restecg', 'fbs', 'exang']

# Cargar el modelo y el vectorizador
with open('heartDisease-model.pck', 'rb') as f:
    dv, svm_model = pickle.load(f)

# Título de la aplicación
st.title('Predicción de Enfermedad Cardíaca')

# Instrucciones
st.write("""
Este modelo predice la presencia de enfermedad cardíaca en pacientes
utilizando características como la edad, colesterol, presión arterial, etc.
Por favor, ingresa los valores para hacer una predicción.
""")

# Ingresar datos del paciente
age = st.number_input('Edad', min_value=1, max_value=100, value=60)
sex = st.selectbox('Sexo', options=[0, 1], format_func=lambda x: 'Masculino' if x == 1 else 'Femenino')
cp = st.selectbox('Tipo de dolor torácico', options=[0, 1, 2, 3], format_func=lambda x: {0: 'Angina típica', 1: 'Angina atípica', 2: 'Dolor no anginoso', 3: 'Asintomático'}[x])
chol = st.number_input('Colesterol en mg/dl', min_value=1, max_value=500, value=200)
thalach = st.number_input('Frecuencia cardiaca máxima alcanzada', min_value=50, max_value=250, value=150)
oldpeak = st.number_input('Depresión del ST inducida por el ejercicio', min_value=0.0, max_value=6.0, value=1.0)
ca = st.selectbox('Número de vasos principales (0-3)', options=[0, 1, 2, 3])
slope = st.selectbox('Pendiente del segmento ST', options=[0, 1, 2], format_func=lambda x: {0: 'Ascendente', 1: 'Horizontal', 2: 'Descendente'}[x])
restecg = st.selectbox('Electrocardiografía en reposo', options=[0, 1, 2], format_func=lambda x: {0: 'Normal', 1: 'Anomalía de onda ST', 2: 'Hipertrofia ventrículo izquierdo'}[x])
fbs = st.selectbox('Glucosa en ayunas mayor a 120 mg/dl', options=[0, 1], format_func=lambda x: 'Sí' if x == 1 else 'No')
exang = st.selectbox('Angina inducida por ejercicio', options=[0, 1], format_func=lambda x: 'Sí' if x == 1 else 'No')

# Crear un diccionario con los datos ingresados
input_data = {
    'age': age,
    'sex': sex,
    'cp': cp,
    'chol': chol,
    'thalach': thalach,
    'oldpeak': oldpeak,
    'ca': ca,
    'slope': slope,
    'restecg': restecg,
    'fbs': fbs,
    'exang': exang
}

# Convertir el diccionario a un DataFrame
input_df = pd.DataFrame([input_data])

# Transformar los datos usando el vectorizador
input_dict = input_df[cat_cols + num_cols].to_dict(orient='records')
input_transformed = dv.transform(input_dict)

# Realizar la predicción
prediction = svm_model.predict(input_transformed)
probability = svm_model.predict_proba(input_transformed)

# Mostrar el resultado
if prediction[0] == 1:
    st.write("**Resultado**: El paciente tiene alta probabilidad de enfermedad cardíaca.")
else:
    st.write("**Resultado**: El paciente tiene baja probabilidad de enfermedad cardíaca.")

# Mostrar la probabilidad (si se seleccionó 'probability=True' en el modelo SVM)
st.write(f"Probabilidad de enfermedad cardíaca: {probability[0][1]:.2f}")
