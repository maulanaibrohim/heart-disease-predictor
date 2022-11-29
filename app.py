#import packages 
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

#set web page
st.set_page_config(
    page_title='Prediksi Sakit Jantung', 
    page_icon=':heartbeat:',
    layout="centered", 
    initial_sidebar_state="auto", 
    menu_items=None)
st.write("""
# Prediksi Probabilitas Sakit Jantung
Aplikasi ini akan memprediksi probabilitas sakit jantung berdasarkan beberapa parameter yang anda berikan!
\n ***Aplikasi ini tidak memberikan diagnosis, kunjungi fasilitas kesehatan terdekat untuk info lebih lanjut!***
\n Silahkan masukkan data anda pada menu di samping! Prediksi akan diperbaharui bersamaan dengan masukkan data yang anda berikan.

""")

#set sidebar
#set header 
st.sidebar.header('Masukkan informasi berikut!')
st.sidebar.markdown("""
*Informasi tersebut digunakan sebagai parameter prediksi!*
""")

#create function and interface to collect data from user
def user_input_features():
    age=st.sidebar.number_input(
        label='Usia (Tahun)', 
        min_value=0, 
        max_value=100,
        value=50,
        step=1
        )
    sex=st.sidebar.selectbox(
        label='Jenis Kelamin',
        options=(
            'Pria', 
            'Wanita')
        )
    rbp=st.sidebar.number_input(
        label='Tekanan Darah Istirahat (mmHg)', 
        min_value=0, 
        max_value=1000,
        value=100,
        step=1
        )
    col=st.sidebar.number_input(
        label='Kadar Kolesterol Total (mg/dL)', 
        min_value=0, 
        max_value=1000,
        value=150,
        step=1
        )
    fbs=st.sidebar.selectbox(
        label='Gula Darah setelah puasa > 120 mg/dL',
        options=(
            'Iya',
            'Tidak')
        )
    mhr=st.sidebar.number_input(
        label='Denyut Jantung Maximum (kali/menit)', 
        min_value=0, 
        max_value=1000,
        value=90,
        step=1
        )
    cpt=st.sidebar.selectbox(
        label='Jenis Nyeri Dada (Angina)',
        options=(
            'Typical Angina', 
            'Atypical Angina',
            'Non-anginal Pain',
            'Asymptomatic')
        )
    ecg=st.sidebar.selectbox(
        label='Hasil Elektrokardiogram',
        options=(
            'Normal', 
            'Gelombang ST-T abnormal',
            'Penebalan sisi kiri dinding otot jantung')
        )
    eia=st.sidebar.selectbox(
        label='Nyeri Dada ketika Beraktivitas',
        options=(
            'Iya',
            'Tidak')
        )
    old=st.sidebar.number_input(
        label='Oldpeak*', 
        min_value=-10.0, 
        max_value=10.0,
        value=0.0,
        step=0.1
        )
    st.sidebar.caption('**Oldpeak merupakan Depresi ST yang diinduksi oleh olahraga relatif terhadap istirahat ("ST" berhubungan dengan posisi pada plot EKG)*')
    sts=st.sidebar.selectbox(
        label='ST-slope*',
        options=(
            'Up-sloping', 
            'Flat',
            'Down-sloping')
        )
    #encode data corresponding to dataset
    st.sidebar.caption('**ST-slope merupakan kemiringan segmen ST latihan puncak*')
    if sex=='Pria':
        sex='M'
    elif sex=='Wanita':
        sex='F'
    
    if fbs=='Iya':
        fbs=1
    elif fbs=='Tidak':
        fbs=0
    
    if cpt=='Typical Angina':
        cpt='TA'
    elif cpt=='Atypical Angina':
        cpt='ATA'
    elif cpt=='Non-anginal Pain':
        cpt='NAP'
    elif cpt=='Asymptomatic':
        cpt='ASY'
    
    if ecg=='Gelombang ST-T abnormal':
        ecg='ST'
    elif ecg=='Penebalan sisi kiri dinding otot jantung':
        ecg='LVH'

    if eia=='Iya':
        eia='Y'
    elif eia=='Tidak':
        eia='N'

    if sts=='Up-sloping':
        sts='Up'
    elif sts=='Down-sloping':
        sts='Down'
    #assign collected data to a variable
    data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': cpt,
        'RestingBP': rbp,
        'Cholesterol': col,
        'FastingBS': fbs,
        'RestingECG': ecg,
        'MaxHR': mhr,
        'ExerciseAngina': eia,
        'Oldpeak':old,
        'ST_Slope': sts
    }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

#read datset
heart_raw = pd.read_csv('heart.csv')
heart = heart_raw.drop(columns=['HeartDisease'])
#merge with user input
df = pd.concat([input_df, heart], axis=0)
#one hot encoding data
df = pd.get_dummies(df)
#get first row of dataset (which mean user input)
df = df[:1]

#read machine learning model
load_clf = pickle.load(open('heart_predict.pkl', 'rb'))

#apply model
predictions = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df).round(decimals=2)

#seting output prediction and interface
st.subheader('Prediksi anda memiliki penyakit jantung')
if predictions < 1:
    st.write('Anda mungkin tidak memiliki penyakit jantung. Tetaplah jaga kesehatan!')
    st.subheader('Probabilitas kebenaran prekdisi')
    st.subheader(f'{prediction_proba.item((0,0))*100}%')
else:
    st.write('Anda sebaiknya mengecek kesehatan jantung anda di fasilitas kesehatan terdekat!')
    st.subheader('Probabilitas kebenaran prekdisi')
    st.subheader(f'{prediction_proba.item((0,1))*100}%')
st.write("***Disclaimer***  Hasil prediksi ini bukanlah diagnosis! Silahkan kunjungi fasilitas kesehatan terdekat!")
