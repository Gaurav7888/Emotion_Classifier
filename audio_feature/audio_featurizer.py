import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import sklearn


def audio_process(songname: str) -> pd.DataFrame:
     pass


def spectrogram_plot(audio_file: str):
    
    y, sr = librosa.load(audio_file, mono=True, duration=5)
    plot = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=13)
    plot = librosa.power_to_db(plot, ref=np.max)
    librosa.display.specshow(plot, y_axis='mel', x_axis='time')
    plt.title('Mel-frequency spectrogram')
    plt.colorbar()
    plt.tight_layout()

def waveplot(audio_file :str):   
    y, sr = librosa.load(audio_file, mono=True, duration=5)
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(y, sr=sr)


def lineplot(audio_file :str):
    x, sr = librosa.load(audio_file, mono=True, duration=5)
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)
    def normalize(x, axis=0):
        return sklearn.preprocessing.minmax_scale(x, axis=axis)
    spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr)[0]
    spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=3)[0]
    spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=4)[0]
    plt.figure(figsize=(15, 9))
    librosa.display.waveplot(x, sr=sr, alpha=0.4)
    plt.plot(t, normalize(spectral_bandwidth_2), color='r')
    plt.plot(t, normalize(spectral_bandwidth_3), color='g')
    plt.plot(t, normalize(spectral_bandwidth_4), color='y')
    plt.legend(('p = 2', 'p = 3', 'p = 4'))
