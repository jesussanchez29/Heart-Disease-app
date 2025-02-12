import streamlit as st
import pickle
import numpy as np

# Cargar el modelo y el Diccionario de características
with open('heartDisease-svm.pck', 'rb') as f:
    dv, model, scaler = pickle.load(f)

# Título de la aplicación
st.title("Predicción de Enfermedades Cardíacas")

# Crear la interfaz de usuario para la entrada de datos
age = st.slider("Edad", min_value=18, max_value=80, value=50)

# Para las columnas categóricas, usamos selectbox
sex = st.selectbox("Sexo", ["Masculino", "Femenino"])
cp = st.selectbox("Tipo de Dolor en el Pecho", ["0", "1", "2", "3"])
fbs = st.selectbox("Nivel de Glucosa en ayunas (fbs)", ["0", "1"])
restecg = st.selectbox("Resultado Electrocardiograma en reposo (restecg)", ["0", "1", "2"])
exang = st.selectbox("Angina inducida por ejercicio (exang)", ["0", "1"])
slope = st.selectbox("Pendiente del ST inducida por el ejercicio (slope)", ["0", "1", "2"])
thal = st.selectbox("Condición de thal", ["3", "6", "7"])

# Para las columnas numéricas, usamos number_input
trestbps = st.number_input("Presión sanguínea en reposo (trestbps)", min_value=94, max_value=200, value=120)
chol = st.number_input("Colesterol", min_value=126, max_value=564, value=240)
thalch = st.number_input("Frecuencia cardíaca máxima (thalch)", min_value=71, max_value=202, value=150)
oldpeak = st.number_input("Depresión del ST inducida por el ejercicio (oldpeak)", min_value=0.0, max_value=6.2, value=1.0)
ca = st.number_input("Número de vasos principales coloreados por fluoroscopia (ca)", min_value=0, max_value=4, value=0)

# Botón para hacer la predicción
if st.button('Hacer Predicción'):
    # Convertir los valores de entrada a un diccionario
    input_data = {
        'age': age,
        'sex': 1 if sex == "Masculino" else 0,
        'cp': int(cp),
        'trestbps': trestbps,
        'chol': chol,
        'fbs': int(fbs),
        'restecg': int(restecg),
        'thalch': thalch,
        'exang': int(exang),
        'oldpeak': oldpeak,
        'slope': int(slope),
        'ca': ca,
        'thal': int(thal)
    }

    # Transformar los datos de entrada utilizando el DictVectorizer
    X_new = dv.transform([input_data])

    # Escalar las características numéricas (aplicar el mismo scaler que durante el entrenamiento)
    num_cols = ['trestbps', 'chol', 'thalch', 'oldpeak', 'ca', 'age']
    num_indices = [i for i, name in enumerate(dv.get_feature_names_out()) if name in num_cols]  # Índices de las características numéricas
    X_new[:, num_indices] = scaler.transform(X_new[:, num_indices])  # Escalar las características numéricas
    st.write(X_new)
    # Realizar la predicción
    y_pred_proba = model.predict_proba(X_new)[0][1]  # Probabilidad de enfermedad cardíaca
    # Mostrar el resultado de la predicción
    if y_pred_proba > 0.5:
        st.write("La persona tiene enfermedad cardíaca.")
    else:
        st.write("La persona no tiene enfermedad cardíaca.")

    # Mostrar la probabilidad de enfermedad
    st.write(f"Probabilidad de tener enfermedad: {y_pred_proba:.2f}")
