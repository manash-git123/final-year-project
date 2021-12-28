import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
import os

parent_dir = "drive/MyDrive/Colab Notebooks/Data/"
a = ["blues","classical","country","disco","hiphop","jazz"]
ar = ["metal","pop","reggae","rock"]
arr = ["classical"]

# generating and storing mel spectograms for all 10 genres
for genre in arr:

  # Creating output folder
  path = os.path.join(parent_dir+"genre", genre)
  os.mkdir(path)

  # Generating Waveplot, Spectograms and Mel Spectograms for first 10 files
  for i in range(10):  

    song = parent_dir+"genres_original/"+genre+"/"+genre+".0000"+str(i)+".wav"

    # gerating waveplot
    scale, sr = librosa.load(song)
    librosa.load(song, sr=None)
    plt.figure(figsize=(16, 5))
    librosa.display.waveplot(scale, sr=sr)

    # generating spectograms
    X = librosa.stft(scale)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(20, 10))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.title('Spectogram')
    plt.colorbar()

    # generating mel spectograms
    y, sr = librosa.load(song)
    melSpec = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=512, n_mels=128)
    melSpec_dB = librosa.power_to_db(melSpec, ref=np.max)
    plt.figure(figsize=(20, 10))
    librosa.display.specshow(melSpec_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')

    # saving in folders
    plt.savefig(parent_dir+"genre/"+genre+"/"+genre+".0000"+str(i)+".png")

  # Generating Waveplot, Spectograms and Mel Spectograms for the next 90 files
  for i in range(10,100):

    song = parent_dir+"genres_original/"+genre+"/"+genre+".000"+str(i)+".wav"

    # gerating waveplot
    scale, sr = librosa.load(song)
    librosa.load(song, sr=None)
    plt.figure(figsize=(16, 5))
    librosa.display.waveplot(scale, sr=sr)

    # generating spectograms
    X = librosa.stft(scale)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(15, 10))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.title('Spectogram')
    plt.colorbar()

    # generating mel spectograms
    y, sr = librosa.load(song)
    melSpec = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=512, n_mels=128)
    melSpec_dB = librosa.power_to_db(melSpec, ref=np.max)
    plt.figure(figsize=(15, 10))
    librosa.display.specshow(melSpec_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')

    # saving in folders
    plt.savefig(parent_dir+"genre/"+genre+"/"+genre+".000"+str(i)+".png")  
