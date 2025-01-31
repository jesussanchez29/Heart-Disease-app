import streamlit as st
import pickle
import numpy as np

# Cargar el modelo y el vectorizador
with open('heartDisease-model.pck', 'rb') as f:
    dv, model = pickle.load(f)

st.title("Predicción de Enfermedades Cardíacas")
st.write("Ingrese los datos del paciente para predecir la probabilidad de enfermedad cardíaca.")

# Crear inputs para las variables
age = st.number_input("Edad", min_value=1, max_value=120, value=50)
trestbps = st.number_input("Presión arterial en reposo (mm Hg)", min_value=50, max_value=250, value=120)
chol = st.number_input("Colesterol sérico (mg/dl)", min_value=100, max_value=600, value=200)
thalach = st.number_input("Frecuencia cardíaca máxima alcanzada", min_value=60, max_value=250, value=150)
oldpeak = st.number_input("Depresión ST inducida por el ejercicio", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
ca = st.number_input("Número de vasos coloreados por fluoroscopía", min_value=0, max_value=4, value=0)

sex = st.selectbox("Sexo", ["0 (Femenino)", "1 (Masculino)"])
cp = st.selectbox("Tipo de dolor torácico", ["0", "1", "2", "3"])
thall = st.selectbox("Thal", ["1", "2", "3"])
slope = st.selectbox("Pendiente del segmento ST", ["0", "1", "2"])
restecg = st.selectbox("Electrocardiograma en reposo", ["0", "1", "2"])
fbs = st.selectbox("Azúcar en sangre en ayunas > 120 mg/dl", ["0", "1"])
exang = st.selectbox("Angina inducida por ejercicio", ["0", "1"])

# Crear diccionario con los valores ingresados
data = {
    "age": age,
    "trestbps": trestbps,
    "chol": chol,
    "thalch": thalach,
    "oldpeak": oldpeak,
    "ca": ca,
    "sex": int(sex[0]),
    "cp": int(cp[0]),
    "thal": int(thall[0]),
    "slope": int(slope[0]),
    "restecg": int(restecg[0]),
    "fbs": int(fbs[0]),
    "exang": int(exang[0])
}

# Transformar los datos usando el vectorizador
data_transformed = dv.transform([data])

# Predecir cuando el usuario presione el botón
if st.button("Predecir"):
    prediction = model.predict([data_transformed[0]])[0]
    st.write(f"Predicción de enfermedad cardíaca: {prediction}")
