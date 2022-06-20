import os
import pickle
import librosa
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



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
path = "/home/gaurav/Desktop/music-genre-classification-using-machine-learning-main/Angry.wav"
extract_features(X, path)
y_pre = model_SVC.predict(X)
le.inverse_transform(y_pre)

print(model_SVC.predict_proba(X))
a = model_SVC.predict_proba(X)
z = ['Angry', 'Happy', 'Neutral', 'Sad']
print(z[np.argmax(a)])

