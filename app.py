import streamlit as st
import pickle
import pandas as pd

# Título de la aplicación
st.title("Predicción de Severidad de Enfermedad Cardíaca")

# Subtítulo
st.markdown("""
Esta aplicación predice la severidad de la enfermedad cardíaca (0 a 4) basada en los datos ingresados.
""")

# Cargar el modelo y el vectorizador
with open('heartDisease-model.pck', 'rb') as f:
    dv, model = pickle.load(f)

# Función para realizar la predicción
def predict_heart_disease(input_data):
    # Convertir el diccionario de entrada en un DataFrame
    input_df = pd.DataFrame([input_data])
    # Vectorizar los datos de entrada
    input_dict = input_df.to_dict(orient='records')
    input_vector = dv.transform(input_dict)
    # Realizar la predicción de probabilidades
    probabilities = model.predict_proba(input_vector)[0]
    # Realizar la predicción de la clase
    prediction = model.predict(input_vector)[0]
    return prediction, probabilities

# Crear un formulario para ingresar los datos
with st.form("input_form"):
    st.header("Ingrese los datos del paciente")

    # Campos de entrada para las características
    age = st.number_input("Edad", min_value=0, max_value=120, value=50)
    sex = st.selectbox("Sexo", options=[0, 1], format_func=lambda x: "Mujer" if x == 0 else "Hombre")
    cp = st.selectbox("Tipo de dolor en el pecho (cp)", options=[0, 1, 2, 3])
    trestbps = st.number_input("Presión arterial en reposo (trestbps)", min_value=0, max_value=200, value=120)
    chol = st.number_input("Colesterol sérico (chol)", min_value=0, max_value=600, value=200)
    fbs = st.selectbox("Azúcar en sangre en ayunas > 120 mg/dl (fbs)", options=[0, 1])
    restecg = st.selectbox("Resultados electrocardiográficos en reposo (restecg)", options=[0, 1, 2])
    thalch = st.number_input("Frecuencia cardíaca máxima alcanzada (thalch)", min_value=0, max_value=250, value=150)
    exang = st.selectbox("Angina inducida por ejercicio (exang)", options=[0, 1])
    oldpeak = st.number_input("Depresión del ST inducida por ejercicio (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Pendiente del segmento ST (slope)", options=[0, 1, 2])
    ca = st.number_input("Número de vasos principales coloreados por fluoroscopia (ca)", min_value=0, max_value=3, value=0)
    thal = st.selectbox("Thal", options=[0, 1, 2, 3])

    # Botón para realizar la predicción
    submitted = st.form_submit_button("Predecir Severidad")

    # Si se presiona el botón, realizar la predicción
    if submitted:
        # Crear un diccionario con los datos ingresados
        input_data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalch': thalch,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }

        # Realizar la predicción
        prediction, probabilities = predict_heart_disease(input_data)

        # Mostrar el resultado
        st.subheader("Resultado de la Predicción")
        st.write(f"La severidad predicha de la enfermedad cardíaca es: **{prediction}**")
        
        # Mostrar las probabilidades para cada clase
        st.subheader("Probabilidades por clase")
        for i, prob in enumerate(probabilities):
            st.write(f"Clase {i}: {prob * 100:.2f}%")

        # Explicación de las clases
        st.markdown("""
        **Nota:** La severidad varía de 0 a 4, donde:
        - 0: Sin enfermedad
        - 1: Enfermedad leve
        - 2: Enfermedad moderada
        - 3: Enfermedad grave
        - 4: Enfermedad muy grave
        """)
