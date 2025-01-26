import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import load

# Cargar el modelo entrenado
model = load('models/heartDisease-model.pck')

# Función para preprocesar los datos de entrada
def preprocess_data(input_data):
    # Variables categóricas
    cat_cols = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    
    # Codificación OneHot
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_cols = encoder.fit_transform(input_data[cat_cols])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(cat_cols))
    
    # Escalar variables numéricas
    num_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    scaler = StandardScaler()
    scaled_cols = scaler.fit_transform(input_data[num_cols])
    scaled_df = pd.DataFrame(scaled_cols, columns=num_cols)

    # Combinar los DataFrames procesados
    processed_df = pd.concat([scaled_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    return processed_df

# Título de la aplicación
st.title('Predicción de Enfermedad Cardíaca')

# Formulario de entrada para el usuario
st.header('Ingrese los datos del paciente')

# Inputs del usuario
age = st.number_input('Edad', min_value=20, max_value=120, value=50)
sex = st.selectbox('Sexo', ['M', 'F'])
cp = st.selectbox('Tipo de dolor en el pecho', [0, 1, 2, 3])
fbs = st.selectbox('Nivel de azúcar en sangre en ayunas', [0, 1])
restecg = st.selectbox('Electrocardiograma en reposo', [0, 1, 2])
exang = st.selectbox('Angina inducida por ejercicio', [0, 1])
slope = st.selectbox('Pendiente del segmento ST', [0, 1, 2])
thal = st.selectbox('Tercer tipo de thalassemia', [3, 6, 7])
trestbps = st.number_input('Presión arterial en reposo', min_value=80, max_value=200, value=120)
chol = st.number_input('Colesterol sérico', min_value=100, max_value=400, value=250)
thalch = st.number_input('Frecuencia máxima de latido alcanzada', min_value=50, max_value=200, value=150)
oldpeak = st.number_input('Depresión del segmento ST inducida por el ejercicio', value=1.0)
ca = st.selectbox('Número de vasos principales coloreados', [0, 1, 2, 3])

# Crear un DataFrame de entrada
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'fbs': [fbs],
    'restecg': [restecg],
    'exang': [exang],
    'slope': [slope],
    'thal': [thal],
    'trestbps': [trestbps],
    'chol': [chol],
    'thalch': [thalch],
    'oldpeak': [oldpeak],
    'ca': [ca]
})

# Preprocesar los datos
processed_data = preprocess_data(input_data)

# Hacer la predicción
prediction = model.predict(processed_data)

# Mostrar el resultado
if prediction == 1:
    st.write('**El paciente tiene una probabilidad de tener enfermedad cardíaca.**')
else:
    st.write('**El paciente no tiene enfermedad cardíaca.**')
