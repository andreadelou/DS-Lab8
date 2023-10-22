import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

predictor = joblib.load("modeloV.sav")

# Título de la aplicación
st.title('Predicciones de Alquiler')

# Crear campos interactivos para ingresar datos
city = st.selectbox("Ciudad", ["Sao Paulo", "Porto Alegre", "Rio de Janeiro", "Campinas", "Belo Horizonte"])
area = st.number_input("Área", min_value=1)
rooms = st.number_input("Habitaciones", min_value=1)
bathroom = st.number_input("Baños", min_value=0)
parking_spaces = st.number_input("Espacios de Estacionamiento", min_value=0)
floor = st.number_input("Piso", min_value=1)
animal = st.selectbox("Permite Animales", ["acept", "not acept"])
furniture = st.selectbox("Amueblado", ["furnished", "not furnished"])
hoa = st.number_input("HOA (R$)", min_value=0)
rent_amount = st.number_input("Renta (R$)", min_value=0)
property_tax = st.number_input("Impuesto a la Propiedad (R$)", min_value=0)
fire_insurance = st.number_input("Seguro contra Incendios (R$)", min_value=0)
total = st.number_input("Total (R$)", min_value=0)

data = pd.read_csv("./data.csv")

# Botón para realizar la predicción
if st.button("Realizar predicción"):
    # Crear un DataFrame con los datos ingresados
    input_data = pd.DataFrame({
        'city': [city],
        'area': [area],
        'rooms': [rooms],
        'bathroom': [bathroom],
        'parking spaces': [parking_spaces],
        'floor': [floor],
        'animal': [animal],
        'furniture': [furniture],
        'hoa (R$)': [hoa],
        'property tax (R$)': [property_tax],
        'fire insurance (R$)': [fire_insurance],
        'total (R$)': [total]
    })

    # Realizar la predicción
    prediction = predictor.predict(input_data)
    
    # Mostrar la predicción
    st.write("Resultado de la Predicción:")
    st.write(prediction[0])

    # Generar gráficas y estadísticas
    st.subheader("Otras Estadísticas sobre el conjunto de datos")
    st.write(data.describe())

    # Gráficos ============================================================

    # Gráfico de cajas de precios de alquiler por ciudad
    st.subheader('Gráfico de Cajas de Precios de Alquiler por Ciudad')
    sns.boxplot(x='city', y='rent amount (R$)', data=data)
    st.pyplot(plt)

    # Gráfico de barras de la distribución de habitaciones por ciudad
    st.subheader('Distribución de Habitaciones por Ciudad')
    sns.countplot(x='rooms', hue='city', data=data)
    st.pyplot(plt)

    # Matriz de correlación
    st.subheader('Matriz de Correlación')
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True)
    st.pyplot(plt)


