import numpy as np
import streamlit as st
import pickle

# Import Model
data = pickle.load(open('data.pkl', 'rb'))
pipe = pickle.load(open('pipe.pkl', 'rb'))

st.title("Predictor Harga Laptop")

# Brand
company = st.selectbox('Brand', data['Company'].unique())

# Tipe
type = st.selectbox('Tipe', data['TypeName'].unique())

# RAM
ram = st.selectbox('RAM', ['4','8','12','16','32','64'])

# Berat
weight = st.number_input('Berat')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# Layar IPS
ips = st.selectbox('Layar IPS', ['No', 'Yes'])

# Screen Size
screen_size = st.number_input('Ukuran Layar')

# Resolution
resolution = st.selectbox('Resolusi Layar', ['1920x1080','1366x768','1600x900','2160x1440','2400x1600','2560x1440','2560x1600','2736x1824','2880x1800','3200x1800','3840x2160'])

# CPU
cpu = st.selectbox('CPU', data['Cpu Brand'].unique())

# HDD
hdd = st.selectbox('HDD', ['0','128','256','512','1024','2048'])

# SSD
ssd = st.selectbox('SSD', ['0','128','256','512','1024'])

# GPU
gpu = st.selectbox('GPU', data['Gpu Brand'].unique())

# OS
os = st.selectbox('Operating System', data['OS'].unique())

if st.button('Prediksi Harga'):
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    # Split nilai ppi dan diubah ke Integer
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,12)
    st.title("Predicted Harga: $ " + str(int(np.exp(pipe.predict(query)[0]))))