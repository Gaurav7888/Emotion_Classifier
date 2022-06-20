import streamlit as st
from audio_feature.audio_featurizer import audio_process, spectrogram_plot, waveplot,lineplot
from models.load_model import model_loader
import numpy as np
import pandas as pd
from pydub import AudioSegment
import os
import pickle
import librosa
import seaborn as sns
import soundfile as sf
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



#

st.sidebar.markdown(
    """<style>body {background-color: #2C3454; background-image: url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRQm90dM3ljhQR0DCcqD8YfPbs2PmKmM2StfA&usqp=CAU'); color:white;}</style><body></body>""",
    unsafe_allow_html=True)
st.markdown(
    """<h1 style='text-align: center; color: white;font-size:60px;margin-top:-50px;'>AUDIO EMOTION CLASSIFIER</h1><h1 style='text-align: center; color: white;font-size:30px;margin-top:-30px;'>Using Machine Learning</h1>""",
    unsafe_allow_html=True)


check = st.sidebar.checkbox('Do you want a demo')



if check:
    rad_test = st.sidebar.radio("Select format of audio file", options=['wav'])

    if rad_test == "wav":
        rad_file = st.sidebar.radio("Select the Audio Type", ["Angry", "Happy","Sad", "Neutral","Abuse","Non-Abusive"])
        if rad_file == "Angry":
            rad = st.sidebar.radio("Choose Options", options=["Predict", "Feature-Plots"])
            st.audio("Angry.wav")
            # rad = st.sidebar.checkbox(label="Do You want to see the spectrogram ?")
            if rad == "Predict":
                if st.button("Classify Audio"):
                    
        

                    def extract_features(x, path):
                        X, sample_rate = sf.read(path, dtype='float32')
                        # Mcc
                        Mcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=47)
                        Mcc = np.mean(Mcc.T, axis=0)
                        # chroma_stft
                        chroma_stft = librosa.feature.chroma_stft(
                            y=X, sr=sample_rate, n_chroma=12, n_fft=4096)
                        chroma_stft = np.mean(chroma_stft.T, axis=0)
                        # chroma_cqt
                        chroma_cqt = librosa.feature.spectral_bandwidth(y=X, sr=sample_rate)
                        chroma_cqt = np.mean(chroma_cqt.T, axis=0)
                        # tonnetz
                        #tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate, chroma=chroma_cqt)
                        # melspectrogram
                        melspectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate)
                        melspectrogram = np.mean(melspectrogram.T, axis=0)
                        # spectral_centroid
                        spectral_centroid = librosa.feature.spectral_centroid(y=X, sr=sample_rate)
                        spectral_centroid = np.mean(spectral_centroid.T, axis=0)
                        # spectral_contrast
                        spectral_contrast = librosa.feature.spectral_contrast(y=X, sr=sample_rate)
                        spectral_contrast = np.mean(spectral_contrast.T, axis=0)
                        feature = np.hstack((Mcc, chroma_stft, chroma_cqt,
                                            melspectrogram, spectral_centroid, spectral_contrast))
                        x.append(feature)


                    def LabelEncoder(arr, le):
                        le.fit(arr)
                        print(le.classes_)
                        Y = le.fit_transform(arr)
                        return Y


                    Y = np.array(['Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                'Angry' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                'Happy' 'Happy' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'])

                    le = preprocessing.LabelEncoder()
                    Y = LabelEncoder(Y, le)
                    print(type(Y))

                    model_SVC = pickle.load(open('SVC.sav', 'rb'))

                    X = []
                    path = "Angry.wav"
                    extract_features(X, path)
                    y_pre = model_SVC.predict(X)
                    le.inverse_transform(y_pre)

                    print(model_SVC.predict_proba(X))
                    a = model_SVC.predict_proba(X)
                    z = ['Angry', 'Happy', 'Neutral', 'Sad']
                    print(z[np.argmax(a)])

                    st.markdown(
                        f"""<h1 style='color:yellow;'>Prediction : <span style='color:white;'>{z[np.argmax(a)]}</span></h1>""",
                        unsafe_allow_html=True)

            elif rad == "Feature-Plots":
                fig = spectrogram_plot("Angry.wav")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.markdown(
                    f"""<h1 style='color:yellow;'>Spectrogram : </h1>""",
                    unsafe_allow_html=True)
                st.pyplot(fig)
                fig = waveplot("Angry.wav")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.markdown(
                    f"""<h1 style='color:yellow;'>WavePlot : </h1>""",
                    unsafe_allow_html=True)
                st.pyplot(fig)
                fig = lineplot("Angry.wav")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.markdown(
                    f"""<h1 style='color:yellow;'>LinePlot : </h1>""",
                    unsafe_allow_html=True)
                st.pyplot(fig)
        
        elif rad_file == "Happy":
            rad = st.sidebar.radio("Choose Options", options=[
                                   "Predict", "Feature-Plots"])
            st.audio("Happy.wav")
            # rad = st.sidebar.checkbox(label="Do You want to see the spectrogram ?")
            if rad == "Predict":
                if st.button("Classify Audio"):

                    def extract_features(x, path):
                        X, sample_rate = sf.read(path, dtype='float32')
                        # Mcc
                        Mcc = librosa.feature.mfcc(
                            y=X, sr=sample_rate, n_mfcc=47)
                        Mcc = np.mean(Mcc.T, axis=0)
                        # chroma_stft
                        chroma_stft = librosa.feature.chroma_stft(
                            y=X, sr=sample_rate, n_chroma=12, n_fft=4096)
                        chroma_stft = np.mean(chroma_stft.T, axis=0)
                        # chroma_cqt
                        chroma_cqt = librosa.feature.spectral_bandwidth(
                            y=X, sr=sample_rate)
                        chroma_cqt = np.mean(chroma_cqt.T, axis=0)
                        # tonnetz
                        #tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate, chroma=chroma_cqt)
                        # melspectrogram
                        melspectrogram = librosa.feature.melspectrogram(
                            y=X, sr=sample_rate)
                        melspectrogram = np.mean(melspectrogram.T, axis=0)
                        # spectral_centroid
                        spectral_centroid = librosa.feature.spectral_centroid(
                            y=X, sr=sample_rate)
                        spectral_centroid = np.mean(
                            spectral_centroid.T, axis=0)
                        # spectral_contrast
                        spectral_contrast = librosa.feature.spectral_contrast(
                            y=X, sr=sample_rate)
                        spectral_contrast = np.mean(
                            spectral_contrast.T, axis=0)
                        feature = np.hstack((Mcc, chroma_stft, chroma_cqt,
                                            melspectrogram, spectral_centroid, spectral_contrast))
                        x.append(feature)

                    def LabelEncoder(arr, le):
                        le.fit(arr)
                        print(le.classes_)
                        Y = le.fit_transform(arr)
                        return Y

                    Y = np.array(['Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                  'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                  'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                  'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                  'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                  'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                  'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                  'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                  'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'])

                    le = preprocessing.LabelEncoder()
                    Y = LabelEncoder(Y, le)
                    print(type(Y))

                    model_SVC = pickle.load(open('SVC.sav', 'rb'))

                    X = []
                    path = "Happy.wav"
                    extract_features(X, path)
                    y_pre = model_SVC.predict(X)
                

                    print(model_SVC.predict_proba(X))
                    a = model_SVC.predict_proba(X)
                    z = ['Angry', 'Happy', 'Neutral', 'Sad']
                    print(z[np.argmax(a)])

                    st.markdown(
                        f"""<h1 style='color:yellow;'>Prediction : <span style='color:white;'>{z[np.argmax(a)]}</span></h1>""",
                        unsafe_allow_html=True)

            elif rad == "Feature-Plots":
                fig = spectrogram_plot("Happy.wav")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.markdown(
                    f"""<h1 style='color:yellow;'>Spectrogram : </h1>""",
                    unsafe_allow_html=True)
                st.pyplot(fig)
                fig = waveplot("Happy.wav")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.markdown(
                    f"""<h1 style='color:yellow;'>WavePlot : </h1>""",
                    unsafe_allow_html=True)
                st.pyplot(fig)
                fig = lineplot("Happy.wav")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.markdown(
                    f"""<h1 style='color:yellow;'>LinePlot : </h1>""",
                    unsafe_allow_html=True)
                st.pyplot(fig)

        elif rad_file == "Sad":
            rad = st.sidebar.radio("Choose Options", options=[
                                   "Predict", "Feature-Plots"])
            st.audio("Sad.wav")
            # rad = st.sidebar.checkbox(label="Do You want to see the spectrogram ?")
            if rad == "Predict":
                if st.button("Classify Audio"):

                    def extract_features(x, path):
                        X, sample_rate = sf.read(path, dtype='float32')
                        # Mcc
                        Mcc = librosa.feature.mfcc(
                            y=X, sr=sample_rate, n_mfcc=47)
                        Mcc = np.mean(Mcc.T, axis=0)
                        # chroma_stft
                        chroma_stft = librosa.feature.chroma_stft(
                            y=X, sr=sample_rate, n_chroma=12, n_fft=4096)
                        chroma_stft = np.mean(chroma_stft.T, axis=0)
                        # chroma_cqt
                        chroma_cqt = librosa.feature.spectral_bandwidth(
                            y=X, sr=sample_rate)
                        chroma_cqt = np.mean(chroma_cqt.T, axis=0)
                        # tonnetz
                        #tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate, chroma=chroma_cqt)
                        # melspectrogram
                        melspectrogram = librosa.feature.melspectrogram(
                            y=X, sr=sample_rate)
                        melspectrogram = np.mean(melspectrogram.T, axis=0)
                        # spectral_centroid
                        spectral_centroid = librosa.feature.spectral_centroid(
                            y=X, sr=sample_rate)
                        spectral_centroid = np.mean(
                            spectral_centroid.T, axis=0)
                        # spectral_contrast
                        spectral_contrast = librosa.feature.spectral_contrast(
                            y=X, sr=sample_rate)
                        spectral_contrast = np.mean(
                            spectral_contrast.T, axis=0)
                        feature = np.hstack((Mcc, chroma_stft, chroma_cqt,
                                            melspectrogram, spectral_centroid, spectral_contrast))
                        x.append(feature)

                    def LabelEncoder(arr, le):
                        le.fit(arr)
                        print(le.classes_)
                        Y = le.fit_transform(arr)
                        return Y

                    Y = np.array(['Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                  'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                  'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                  'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                  'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                  'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                  'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                  'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                  'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'])

                    le = preprocessing.LabelEncoder()
                    Y = LabelEncoder(Y, le)
                    print(type(Y))

                    model_SVC = pickle.load(open('SVC.sav', 'rb'))

                    X = []
                    path = "Sad.wav"
                    extract_features(X, path)
                    y_pre = model_SVC.predict(X)

                    print(model_SVC.predict_proba(X))
                    a = model_SVC.predict_proba(X)
                    z = ['Angry', 'Happy', 'Neutral', 'Sad']
                    print(z[np.argmax(a)])

                    st.markdown(
                        f"""<h1 style='color:yellow;'>Prediction : <span style='color:white;'>{z[np.argmax(a)]}</span></h1>""",
                        unsafe_allow_html=True)

            elif rad == "Feature-Plots":
                fig = spectrogram_plot("Sad.wav")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.markdown(
                    f"""<h1 style='color:yellow;'>Spectrogram : </h1>""",
                    unsafe_allow_html=True)
                st.pyplot(fig)

                fig = waveplot("Sad.wav")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.markdown(
                    f"""<h1 style='color:yellow;'>WavePlot : </h1>""",
                    unsafe_allow_html=True)
                st.pyplot(fig)
                fig = lineplot("Sad.wav")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.markdown(
                    f"""<h1 style='color:yellow;'>LinePlot : </h1>""",
                    unsafe_allow_html=True)
                st.pyplot(fig)

        elif rad_file == "Neutral":
            rad = st.sidebar.radio("Choose Options", options=[
                                   "Predict", "Feature-Plots"])
            st.audio("Neutral.wav")
            # rad = st.sidebar.checkbox(label="Do You want to see the spectrogram ?")
            if rad == "Predict":
                if st.button("Classify Audio"):

                    def extract_features(x, path):
                        X, sample_rate = sf.read(path, dtype='float32')
                        # Mcc
                        Mcc = librosa.feature.mfcc(
                            y=X, sr=sample_rate, n_mfcc=47)
                        Mcc = np.mean(Mcc.T, axis=0)
                        # chroma_stft
                        chroma_stft = librosa.feature.chroma_stft(
                            y=X, sr=sample_rate, n_chroma=12, n_fft=4096)
                        chroma_stft = np.mean(chroma_stft.T, axis=0)
                        # chroma_cqt
                        chroma_cqt = librosa.feature.spectral_bandwidth(
                            y=X, sr=sample_rate)
                        chroma_cqt = np.mean(chroma_cqt.T, axis=0)
                        # tonnetz
                        #tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate, chroma=chroma_cqt)
                        # melspectrogram
                        melspectrogram = librosa.feature.melspectrogram(
                            y=X, sr=sample_rate)
                        melspectrogram = np.mean(melspectrogram.T, axis=0)
                        # spectral_centroid
                        spectral_centroid = librosa.feature.spectral_centroid(
                            y=X, sr=sample_rate)
                        spectral_centroid = np.mean(
                            spectral_centroid.T, axis=0)
                        # spectral_contrast
                        spectral_contrast = librosa.feature.spectral_contrast(
                            y=X, sr=sample_rate)
                        spectral_contrast = np.mean(
                            spectral_contrast.T, axis=0)
                        feature = np.hstack((Mcc, chroma_stft, chroma_cqt,
                                            melspectrogram, spectral_centroid, spectral_contrast))
                        x.append(feature)

                    def LabelEncoder(arr, le):
                        le.fit(arr)
                        print(le.classes_)
                        Y = le.fit_transform(arr)
                        return Y

                    Y = np.array(['Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry' 'Angry'
                                  'Angry' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy' 'Happy'
                                  'Happy' 'Happy' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Neutral'
                                  'Neutral' 'Neutral' 'Neutral' 'Neutral' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                  'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                  'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                  'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                  'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                  'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                  'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                  'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'
                                  'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad' 'Sad'])

                    le = preprocessing.LabelEncoder()
                    Y = LabelEncoder(Y, le)
                    print(type(Y))

                    model_SVC = pickle.load(open('SVC.sav', 'rb'))

                    X = []
                    path = "Neutral.wav"
                    extract_features(X, path)
                    y_pre = model_SVC.predict(X)

                    print(model_SVC.predict_proba(X))
                    a = model_SVC.predict_proba(X)
                    z = ['Angry', 'Happy', 'Neutral', 'Sad']
                    print(z[np.argmax(a)])

                    st.markdown(
                        f"""<h1 style='color:yellow;'>Prediction : <span style='color:white;'>{z[np.argmax(a)]}</span></h1>""",
                        unsafe_allow_html=True)

            elif rad == "Feature-Plots":
                fig = spectrogram_plot("Neutral.wav")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.markdown(
                    f"""<h1 style='color:yellow;'>Spectrogram : </h1>""",
                    unsafe_allow_html=True)
                st.pyplot(fig)

                fig = waveplot("Neutral.wav")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.markdown(
                    f"""<h1 style='color:yellow;'>WavePlot : </h1>""",
                    unsafe_allow_html=True)
                st.pyplot(fig)
                fig = lineplot("Neutral.wav")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.markdown(
                    f"""<h1 style='color:yellow;'>LinePlot : </h1>""",
                    unsafe_allow_html=True)
                st.pyplot(fig)

        if rad_file == "Abuse":
            rad = st.sidebar.radio("Choose Options", options=[
                                   "Predict", "Feature-Plots"])
            st.audio("Abuse.wav")
            # rad = st.sidebar.checkbox(label="Do You want to see the spectrogram ?")
            if rad == "Predict":
                if st.button("Classify Audio"):

                    def extract_features(x, path):
                        X, sample_rate = sf.read(path, dtype='float32')
                        # Mcc
                        Mcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=47)
                        Mcc = np.mean(Mcc.T, axis=0)
                        # chroma_stft
                        chroma_stft = librosa.feature.chroma_stft(
                            y=X, sr=sample_rate, n_chroma=12, n_fft=4096)
                        chroma_stft = np.mean(chroma_stft.T, axis=0)
                        # chroma_cqt
                        chroma_cqt = librosa.feature.spectral_bandwidth(y=X, sr=sample_rate)
                        chroma_cqt = np.mean(chroma_cqt.T, axis=0)
                        # tonnetz
                        tonnetz = librosa.feature.tonnetz(
                            y=librosa.effects.harmonic(X), sr=sample_rate, chroma=chroma_cqt)
                        # melspectrogram
                        melspectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate)
                        melspectrogram = np.mean(melspectrogram.T, axis=0)
                        # spectral_centroid
                        spectral_centroid = librosa.feature.spectral_centroid(y=X, sr=sample_rate)
                        spectral_centroid = np.mean(spectral_centroid.T, axis=0)
                        # spectral_contrast
                        spectral_contrast = librosa.feature.spectral_contrast(y=X, sr=sample_rate)
                        spectral_contrast = np.mean(spectral_contrast.T, axis=0)
                        feature = np.hstack((Mcc, chroma_stft, chroma_cqt, melspectrogram,
                                            spectral_centroid, spectral_contrast, tonnetz))
                        x.append(feature)


                    def LabelEncoder(arr, le):
                        le.fit(arr)
                        print(le.classes_)
                        Y = le.fit_transform(arr)
                        return Y


                    Y = np.array(['No' 'Yes' 'Yes' 'No' 'Yes' 'Yes' 'Yes' 'Yes' 'Yes' 'Yes' 'No' 'No' 'Yes'
                                    'No' 'No' 'Yes' 'Yes' 'Yes' 'No' 'No' 'Yes' 'No' 'No' 'Yes' 'No' 'No'
                                    'No' 'No' 'Yes' 'Yes' 'Yes' 'No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'No'
                                    'No' 'Yes' 'No' 'No' 'No' 'Yes' 'Yes' 'No' 'No' 'No' 'Yes' 'Yes' 'No'
                                    'No' 'No' 'No' 'Yes' 'Yes' 'No' 'Yes' 'No' 'No' 'Yes' 'No' 'No' 'No' 'No'
                                    'Yes' 'Yes' 'Yes' 'Yes' 'No' 'No' 'Yes' 'Yes' 'Yes' 'Yes' 'Yes' 'No' 'No'
                                    'Yes' 'No' 'Yes' 'No' 'No' 'No' 'Yes' 'No' 'No' 'No' 'Yes' 'No' 'No'
                                    'Yes' 'No' 'No' 'Yes' 'No' 'Yes' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No'
                                    'No' 'Yes' 'Yes' 'No' 'Yes' 'Yes' 'Yes' 'Yes' 'No' 'No' 'No' 'No' 'No'
                                    'Yes' 'Yes' 'Yes' 'Yes' 'No' 'Yes' 'No' 'No' 'Yes' 'No' 'No' 'No' 'No'
                                    'No' 'No' 'No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'Yes' 'No' 'Yes' 'No' 'No'
                                    'No' 'No' 'Yes' 'Yes' 'Yes' 'No' 'Yes' 'No' 'Yes' 'No' 'No' 'No' 'Yes'
                                    'No' 'Yes' 'No' 'No' 'Yes' 'No' 'No' 'No' 'Yes' 'No' 'Yes' 'No' 'Yes'
                                    'No' 'No' 'Yes' 'No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'Yes' 'Yes' 'No' 'No'
                                    'No' 'No' 'No' 'No' 'No' 'Yes' 'No' 'Yes' 'Yes' 'Yes' 'Yes' 'No' 'No'
                                    'No' 'No' 'No' 'Yes' 'No' 'No' 'Yes' 'No' 'Yes' 'No' 'No' 'No' 'No' 'No'
                                    'Yes' 'Yes' 'No' 'No' 'Yes' 'No' 'No' 'No' 'No' 'No' 'No' 'Yes' 'No' 'No'
                                    'Yes' 'No' 'No' 'No' 'Yes' 'No' 'Yes' 'No' 'No' 'No' 'No' 'No' 'No' 'No'
                                    'No' 'Yes' 'Yes' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'No' 'No' 'Yes' 'Yes'
                                    'Yes' 'Yes' 'No' 'Yes' 'Yes' 'No' 'No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No'
                                    'Yes' 'No' 'No' 'No' 'Yes' 'No' 'No' 'No' 'No' 'Yes' 'Yes' 'Yes' 'No'
                                    'No' 'Yes' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'Yes' 'No' 'Yes' 'No'
                                    'No' 'Yes' 'No' 'Yes' 'No' 'No' 'Yes' 'No' 'No' 'Yes' 'No' 'No' 'Yes'
                                    'Yes' 'Yes' 'No' 'No' 'No' 'Yes' 'Yes' 'No' 'No' 'Yes' 'No' 'Yes' 'No'
                                    'No' 'No' 'No' 'Yes' 'No' 'Yes' 'Yes' 'No' 'No' 'Yes' 'No' 'No' 'No' 'No'
                                    'No' 'No' 'Yes' 'Yes' 'No' 'No' 'No' 'No' 'Yes' 'No' 'No' 'No' 'Yes' 'No'
                                    'Yes' 'Yes' 'No' 'Yes' 'No' 'Yes' 'No' 'Yes' 'No' 'Yes' 'Yes' 'No' 'Yes'
                                    'Yes' 'Yes' 'Yes' 'No' 'No' 'Yes' 'Yes' 'Yes' 'Yes' 'No' 'Yes' 'No' 'No'
                                    'No' 'Yes' 'No' 'No' 'No' 'Yes' 'No' 'No' 'No' 'No' 'No' 'Yes' 'No' 'Yes'
                                    'No' 'No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'No' 'No' 'Yes' 'Yes' 'Yes'
                                    'No' 'Yes' 'No' 'No' 'No' 'No' 'Yes' 'No' 'No' 'No' 'Yes' 'No' 'No' 'Yes'
                                    'Yes' 'Yes' 'Yes' 'Yes' 'No' 'No' 'No' 'Yes' 'No' 'No' 'No' 'No' 'Yes'
                                    'No' 'No' 'No' 'Yes' 'Yes' 'No' 'Yes' 'No' 'No' 'No' 'No' 'Yes' 'Yes'
                                    'No' 'No' 'No' 'No' 'Yes' 'No' 'Yes' 'Yes' 'Yes' 'Yes' 'Yes' 'No' 'No'
                                    'Yes' 'Yes' 'No' 'Yes' 'Yes' 'Yes' 'Yes' 'No' 'Yes' 'Yes' 'No' 'No' 'Yes'
                                    'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'Yes'
                                    'Yes' 'Yes' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'No' 'No' 'No' 'No' 'No'
                                    'No' 'Yes' 'Yes' 'Yes' 'No' 'No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'Yes'
                                    'Yes' 'No' 'Yes' 'No' 'Yes' 'No' 'No' 'No' 'No' 'No' 'No' 'Yes' 'Yes'
                                    'No' 'No' 'No' 'Yes' 'Yes' 'Yes' 'Yes' 'Yes' 'Yes' 'No' 'Yes' 'Yes' 'No'
                                    'No' 'No' 'No' 'No' 'Yes' 'No' 'No' 'No' 'No' 'No' 'Yes' 'No' 'No' 'No'
                                    'Yes' 'No' 'No' 'No' 'No'])

                    le = preprocessing.LabelEncoder()
                    Y = LabelEncoder(Y, le)

                    Y = np.array(['No' 'Yes'])

                    model = pickle.load(open('SVC_Model_abuse.sav', 'rb'))

                    X = []
                    path = "Abuse.wav"
                    extract_features(X, path)
                    y_pre = model.predict(X)
                    if(y_pre==1):
                        Result = "Abuse Detected"
                        st.markdown(
                            f"""<h1 style='color:yellow;'>Prediction : <span style='color:white;'>{Result}</span></h1>""",
                            unsafe_allow_html=True)
                    else:
                        Result = "Abuse Not Detected"
                        st.markdown(
                            f"""<h1 style='color:yellow;'>Prediction : <span style='color:white;'>{Result}</span></h1>""",
                            unsafe_allow_html=True)

            elif rad == "Feature-Plots":
                fig = spectrogram_plot("Abuse.wav")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.markdown(
                    f"""<h1 style='color:yellow;'>Spectrogram : </h1>""",
                    unsafe_allow_html=True)
                st.pyplot(fig)

                fig = waveplot("Abuse.wav")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.markdown(
                    f"""<h1 style='color:yellow;'>WavePlot : </h1>""",
                    unsafe_allow_html=True)
                st.pyplot(fig)
                fig = lineplot("Abuse.wav")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.markdown(
                    f"""<h1 style='color:yellow;'>LinePlot : </h1>""",
                    unsafe_allow_html=True)
                st.pyplot(fig)


        if rad_file == "Non-Abusive":
            rad = st.sidebar.radio("Choose Options", options=[
                                   "Predict", "Feature-Plots"])
            st.audio("No-Abuse.wav")
            # rad = st.sidebar.checkbox(label="Do You want to see the spectrogram ?")
            if rad == "Predict":
                if st.button("Classify Audio"):

                    def extract_features(x, path):
                        X, sample_rate = sf.read(path, dtype='float32')
                        # Mcc
                        Mcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=47)
                        Mcc = np.mean(Mcc.T, axis=0)
                        # chroma_stft
                        chroma_stft = librosa.feature.chroma_stft(
                            y=X, sr=sample_rate, n_chroma=12, n_fft=4096)
                        chroma_stft = np.mean(chroma_stft.T, axis=0)
                        # chroma_cqt
                        chroma_cqt = librosa.feature.spectral_bandwidth(y=X, sr=sample_rate)
                        chroma_cqt = np.mean(chroma_cqt.T, axis=0)
                        # tonnetz
                        tonnetz = librosa.feature.tonnetz(
                            y=librosa.effects.harmonic(X), sr=sample_rate, chroma=chroma_cqt)
                        # melspectrogram
                        melspectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate)
                        melspectrogram = np.mean(melspectrogram.T, axis=0)
                        # spectral_centroid
                        spectral_centroid = librosa.feature.spectral_centroid(y=X, sr=sample_rate)
                        spectral_centroid = np.mean(spectral_centroid.T, axis=0)
                        # spectral_contrast
                        spectral_contrast = librosa.feature.spectral_contrast(y=X, sr=sample_rate)
                        spectral_contrast = np.mean(spectral_contrast.T, axis=0)
                        feature = np.hstack((Mcc, chroma_stft, chroma_cqt, melspectrogram,
                                            spectral_centroid, spectral_contrast, tonnetz))
                        x.append(feature)


                    def LabelEncoder(arr, le):
                        le.fit(arr)
                        print(le.classes_)
                        Y = le.fit_transform(arr)
                        return Y


                    Y = np.array(['No' 'Yes' 'Yes' 'No' 'Yes' 'Yes' 'Yes' 'Yes' 'Yes' 'Yes' 'No' 'No' 'Yes'
                                    'No' 'No' 'Yes' 'Yes' 'Yes' 'No' 'No' 'Yes' 'No' 'No' 'Yes' 'No' 'No'
                                    'No' 'No' 'Yes' 'Yes' 'Yes' 'No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'No'
                                    'No' 'Yes' 'No' 'No' 'No' 'Yes' 'Yes' 'No' 'No' 'No' 'Yes' 'Yes' 'No'
                                    'No' 'No' 'No' 'Yes' 'Yes' 'No' 'Yes' 'No' 'No' 'Yes' 'No' 'No' 'No' 'No'
                                    'Yes' 'Yes' 'Yes' 'Yes' 'No' 'No' 'Yes' 'Yes' 'Yes' 'Yes' 'Yes' 'No' 'No'
                                    'Yes' 'No' 'Yes' 'No' 'No' 'No' 'Yes' 'No' 'No' 'No' 'Yes' 'No' 'No'
                                    'Yes' 'No' 'No' 'Yes' 'No' 'Yes' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No'
                                    'No' 'Yes' 'Yes' 'No' 'Yes' 'Yes' 'Yes' 'Yes' 'No' 'No' 'No' 'No' 'No'
                                    'Yes' 'Yes' 'Yes' 'Yes' 'No' 'Yes' 'No' 'No' 'Yes' 'No' 'No' 'No' 'No'
                                    'No' 'No' 'No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'Yes' 'No' 'Yes' 'No' 'No'
                                    'No' 'No' 'Yes' 'Yes' 'Yes' 'No' 'Yes' 'No' 'Yes' 'No' 'No' 'No' 'Yes'
                                    'No' 'Yes' 'No' 'No' 'Yes' 'No' 'No' 'No' 'Yes' 'No' 'Yes' 'No' 'Yes'
                                    'No' 'No' 'Yes' 'No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'Yes' 'Yes' 'No' 'No'
                                    'No' 'No' 'No' 'No' 'No' 'Yes' 'No' 'Yes' 'Yes' 'Yes' 'Yes' 'No' 'No'
                                    'No' 'No' 'No' 'Yes' 'No' 'No' 'Yes' 'No' 'Yes' 'No' 'No' 'No' 'No' 'No'
                                    'Yes' 'Yes' 'No' 'No' 'Yes' 'No' 'No' 'No' 'No' 'No' 'No' 'Yes' 'No' 'No'
                                    'Yes' 'No' 'No' 'No' 'Yes' 'No' 'Yes' 'No' 'No' 'No' 'No' 'No' 'No' 'No'
                                    'No' 'Yes' 'Yes' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'No' 'No' 'Yes' 'Yes'
                                    'Yes' 'Yes' 'No' 'Yes' 'Yes' 'No' 'No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No'
                                    'Yes' 'No' 'No' 'No' 'Yes' 'No' 'No' 'No' 'No' 'Yes' 'Yes' 'Yes' 'No'
                                    'No' 'Yes' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'Yes' 'No' 'Yes' 'No'
                                    'No' 'Yes' 'No' 'Yes' 'No' 'No' 'Yes' 'No' 'No' 'Yes' 'No' 'No' 'Yes'
                                    'Yes' 'Yes' 'No' 'No' 'No' 'Yes' 'Yes' 'No' 'No' 'Yes' 'No' 'Yes' 'No'
                                    'No' 'No' 'No' 'Yes' 'No' 'Yes' 'Yes' 'No' 'No' 'Yes' 'No' 'No' 'No' 'No'
                                    'No' 'No' 'Yes' 'Yes' 'No' 'No' 'No' 'No' 'Yes' 'No' 'No' 'No' 'Yes' 'No'
                                    'Yes' 'Yes' 'No' 'Yes' 'No' 'Yes' 'No' 'Yes' 'No' 'Yes' 'Yes' 'No' 'Yes'
                                    'Yes' 'Yes' 'Yes' 'No' 'No' 'Yes' 'Yes' 'Yes' 'Yes' 'No' 'Yes' 'No' 'No'
                                    'No' 'Yes' 'No' 'No' 'No' 'Yes' 'No' 'No' 'No' 'No' 'No' 'Yes' 'No' 'Yes'
                                    'No' 'No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'No' 'No' 'Yes' 'Yes' 'Yes'
                                    'No' 'Yes' 'No' 'No' 'No' 'No' 'Yes' 'No' 'No' 'No' 'Yes' 'No' 'No' 'Yes'
                                    'Yes' 'Yes' 'Yes' 'Yes' 'No' 'No' 'No' 'Yes' 'No' 'No' 'No' 'No' 'Yes'
                                    'No' 'No' 'No' 'Yes' 'Yes' 'No' 'Yes' 'No' 'No' 'No' 'No' 'Yes' 'Yes'
                                    'No' 'No' 'No' 'No' 'Yes' 'No' 'Yes' 'Yes' 'Yes' 'Yes' 'Yes' 'No' 'No'
                                    'Yes' 'Yes' 'No' 'Yes' 'Yes' 'Yes' 'Yes' 'No' 'Yes' 'Yes' 'No' 'No' 'Yes'
                                    'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'Yes'
                                    'Yes' 'Yes' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'No' 'No' 'No' 'No' 'No'
                                    'No' 'Yes' 'Yes' 'Yes' 'No' 'No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'Yes'
                                    'Yes' 'No' 'Yes' 'No' 'Yes' 'No' 'No' 'No' 'No' 'No' 'No' 'Yes' 'Yes'
                                    'No' 'No' 'No' 'Yes' 'Yes' 'Yes' 'Yes' 'Yes' 'Yes' 'No' 'Yes' 'Yes' 'No'
                                    'No' 'No' 'No' 'No' 'Yes' 'No' 'No' 'No' 'No' 'No' 'Yes' 'No' 'No' 'No'
                                    'Yes' 'No' 'No' 'No' 'No'])

                    le = preprocessing.LabelEncoder()
                    Y = LabelEncoder(Y, le)

                    Y = np.array(['No' 'Yes'])

                    model = pickle.load(open('SVC_Model_abuse.sav', 'rb'))

                    X = []
                    path = "No-Abuse.wav"
                    extract_features(X, path)
                    y_pre = model.predict(X)
                    if(y_pre==1):
                        Result = "Abuse Detected"
                        st.markdown(
                            f"""<h1 style='color:yellow;'>Prediction : <span style='color:white;'>{Result}</span></h1>""",
                            unsafe_allow_html=True)
                    else:
                        Result = "Abuse Not Detected"
                        st.markdown(
                            f"""<h1 style='color:yellow;'>Prediction : <span style='color:white;'>{Result}</span></h1>""",
                            unsafe_allow_html=True)

            elif rad == "Feature-Plots":
                fig = spectrogram_plot("No-Abuse.wav")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.markdown(
                    f"""<h1 style='color:yellow;'>Spectrogram : </h1>""",
                    unsafe_allow_html=True)
                st.pyplot(fig)

                fig = waveplot("No-Abuse.wav")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.markdown(
                    f"""<h1 style='color:yellow;'>WavePlot : </h1>""",
                    unsafe_allow_html=True)
                st.pyplot(fig)

                fig = lineplot("No-Abuse.wav")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.markdown(
                    f"""<h1 style='color:yellow;'>LinePlot : </h1>""",
                    unsafe_allow_html=True)
                st.pyplot(fig)
