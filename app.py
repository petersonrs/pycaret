#carregando as bibliotecas
from operator import index
import pandas as pd
import streamlit as st
from pycaret.regression import *

#carregando o modelo treinado.
model = load_model('CatBoost_Regressor')

from pycaret.datasets import get_data
dataset = get_data('diamond')

# título
st.title("Previsão Valor do Diamante")

# subtítulo
st.markdown("Este é um Data App utilizado para prever valor de diamante.")

#barra esquerda com as definições de entrada
st.sidebar.subheader("Defina os atributos para predição")

#	Report
# mapeando dados do usuário para cada atributo
Carat_Weight = st.sidebar.number_input("Carat Weight", value=dataset["Carat Weight"].mean())
Carat_Weight = round(Carat_Weight,2)

Cut = st.sidebar.selectbox("Cut", ("Signature-Ideal", "Ideal", "Very Good", "Good", "Fair"))
Color = st.sidebar.selectbox("Color", ("D", "E", "F", "G", "H", "I"))
Clarity = st.sidebar.selectbox("Clarity", ("F", "IF", "VVS1 or VVS2", "VS1 or VS2", "SI1"))
Polish = st.sidebar.selectbox("Polish", ("ID", "EX", "VG", "G"))
Symmetry = st.sidebar.selectbox("Symmetry", ("ID", "EX", "VG", "G"))
Report = st.sidebar.selectbox("Report", ("AGSL", "GIA"))

# inserindo um botão na tela
btn_predict = st.sidebar.button("Realizar Classificação")
# verifica se o botão foi acionado
if btn_predict:
    data_teste = pd.DataFrame()
    data_teste["Carat Weight"] = [Carat_Weight]
    data_teste["Cut"] = [Cut]
    data_teste["Color"] = [Color]
    data_teste["Clarity"] = [Clarity]
    data_teste["Polish"] = [Polish]
    data_teste["Symmetry"] = [Symmetry]
    data_teste["Report"] = [Report]

    # #realiza a predição
    result = predict_model(model, data=data_teste)

    st.write(result)