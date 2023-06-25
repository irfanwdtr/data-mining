# import library yang diperlukan

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from sklearn.naive_bayes import GaussianNB
import plotly.express as px
import matplotlib as plt
import seaborn as sns  

# menampilkan teks HTML pada streamlit
st.write("""
# Klasifikasi Penjualan (Web App)
Aplikasi berbasis web untuk mengklasifikasikan jenis pembelian produk digital di **Diantorostore**.
""")

# menampilkan gambar
img = Image.open('gambar1.png')
img = img.resize((700, 418))
st.image(img, use_column_width=False)

# st.sidebar digunakan untuk menampilkan elemen pada sidebar yang terletak di sisi kiri antarmuka aplikasi web.

st.sidebar.header('Parameter Inputan')

# upload file csv untuk parameter inputan
upload_file = st.sidebar.file_uploader('Upload file csv anda', type=['csv'])
if upload_file is not None:
    inputan = pd.read_csv(upload_file)

else:
    def input_user():
        jenis_kelamin = st.sidebar.selectbox('Kelamin', ('Pria', 'Wanita'))
        durasi_options = [1, 4]
        durasi = st.sidebar.radio('Durasi (bulan)', durasi_options)
        jumlah = st.sidebar.text_input('Jumlah', '1')
        usia = st.sidebar.slider('Usia Pembeli', 18, 50, 18)
        diskon_shopee = st.sidebar.slider('Diskon shopee', 0, 3000, 500, step=500)
        diskon_penjual = st.sidebar.radio('Diskon penjual  (max 500)', (0,500))
        cashback = st.sidebar.slider('Cashback', 0, 3000, 0, step=500)

# Setelah mengumpulkan input dari pengguna melalui elemen-elemen antarmuka, nilai-nilai tersebut kemudian disimpan dalam dictionary data

        data = {
            'durasi': durasi,
            'jumlah': int(jumlah),
            'usia': usia,
            'jenis_kelamin': jenis_kelamin,
            'diskon_shopee': diskon_shopee,
            'diskon_penjual': diskon_penjual,
            'cashback': cashback,
        }

# mengubah data input yang telah dikumpulkan dari pengguna menjadi sebuah DataFrame menggunakan library pandas.
        fitur = pd.DataFrame(data, index=[0])
        return fitur

    inputan = input_user()

#  untuk memuat data penjualan dari file CSV ke dalam sebuah DataFrame menggunakan library pandas.
penjualan_raw = pd.read_csv('penjualan50.csv')
penjualan = penjualan_raw.drop(columns=['nama_produk']) 

df = pd.concat([inputan, penjualan], axis=0)


# encoding pada kolom 'jenis_kelamin' dalam DataFrame df.
encode = ['jenis_kelamin']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

df = df[:1]


st.subheader('Parameter Inputan')
if upload_file is not None:
    st.write(df)
else:
    st.write('Menunggu file csv untuk diupload. Saat ini menggunakan sampel inputan (seperti tampilan di bawah).')
    st.write(df)

# kode ini bertujuan untuk menggunakan model yang telah disimpan sebelumnya untuk melakukan prediksi pada data input yang ada dalam DataFrame df.
load_model = pickle.load(open('modelNBC_penjualan.pkl', 'rb'))
prediksi = load_model.predict(df)
prediksi_proba = load_model.predict_proba(df)

# bertujuan untuk menampilkan hasil prediksi jenis pembelian produk digital yang dihasilkan oleh model.

st.subheader('Keterangan Label Kelas')
jenis_penjualan = np.array(['spotify', 'youtube', 'netflix'])
st.write(jenis_penjualan)

st.subheader('Hasil prediksi jenis pembelian produk digital')
st.write(jenis_penjualan[prediksi])


# Menambahkan gambar visual
if prediksi[0] == 0:
    img_spotify = Image.open('spotify.png')
    st.image(img_spotify, caption='Spotify', width=200)
elif prediksi[0] == 1:
    img_youtube = Image.open('youtube.png')
    st.image(img_youtube, caption='YouTube', width=200)
elif prediksi[0] == 2:
    img_netflix = Image.open('netflix.png')
    st.image(img_netflix, caption='Netflix', width=200)

# menampilkan probabilitas hasil prediksi berupa angka
st.subheader('Probabilitas hasil prediksi')
st.write(prediksi_proba)

# Menampilkan grafik probabilitas menggunakan library pylotly
fig = px.bar(x=jenis_penjualan, y=prediksi_proba[0])
fig.update_layout(
    title='Grafik Probabilitas hasil prediksi',
    xaxis_title='Jenis',
    yaxis_title='Probabilitas'
)
st.plotly_chart(fig)
