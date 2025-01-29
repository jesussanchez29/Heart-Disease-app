import streamlit as st
import pickle
import numpy as np

# Cargar el modelo entrenado
with open("heartDisease-model.pck", "rb") as f:
    model = pickle.load(f)

# Función para interpretar la predicción
def interpretar_prediccion(valor):
    interpretacion = {
        0: "Sin enfermedad cardíaca (No se detecta estrechamiento significativo en las arterias).",
        1: "Enfermedad cardíaca leve (Un vaso principal afectado).",
        2: "Enfermedad cardíaca moderada (Dos vasos principales afectados).",
        3: "Enfermedad cardíaca grave (Tres vasos principales afectados).",
        4: "Enfermedad cardíaca muy grave (Cuatro vasos principales afectados).",
    }
    return interpretacion.get(valor, "Valor desconocido")

# Interfaz de Streamlit
st.title("Predicción de Enfermedad Cardíaca")

# Entrada de datos del usuario
age = st.number_input("Edad", min_value=20, max_value=100, step=1)
sex = st.selectbox("Sexo", ["Hombre", "Mujer"])
cp = st.number_input("Tipo de dolor en el pecho (cp)", min_value=0, max_value=3, step=1)
trestbps = st.number_input("Presión arterial en reposo (trestbps)", min_value=90, max_value=200, step=1)
chol = st.number_input("Colesterol sérico (chol)", min_value=100, max_value=600, step=1)
fbs = st.selectbox("Glucosa en ayunas > 120 mg/dl (fbs)", [0, 1])
restecg = st.number_input("Resultados del electrocardiograma en reposo (restecg)", min_value=0, max_value=2, step=1)
thalach = st.number_input("Frecuencia cardíaca máxima alcanzada (thalach)", min_value=70, max_value=250, step=1)
exang = st.selectbox("Angina inducida por ejercicio (exang)", [0, 1])
oldpeak = st.number_input("Depresión ST inducida por el ejercicio (oldpeak)", min_value=0.0, max_value=6.2, step=0.1)
slope = st.number_input("Pendiente del segmento ST (slope)", min_value=0, max_value=2, step=1)
ca = st.number_input("Número de vasos principales coloreados por fluoroscopia (ca)", min_value=0, max_value=4, step=1)
thal = st.number_input("Tipo de talasemia (thal)", min_value=0, max_value=3, step=1)

# Convertir valores categóricos
sex = 1 if sex == "Hombre" else 0

# Predicción
if st.button("Predecir"):
    entrada = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    resultado = model.predict(entrada)[0]
    st.subheader("Resultado de la predicción")
    st.write(interpretar_prediccion(resultado))
