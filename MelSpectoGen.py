import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
import os

parent_dir = "drive/MyDrive/Colab Notebooks/Data/"
arr = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

for genre in arr:

  # Create output folder
  path = os.path.join(parent_dir+"mel_spectograms", genre)
  os.mkdir(path)

  # Generate Mel Spectograms
  for i in range(10):  
    song = parent_dir+"genres_original/"+genre+"/"+genre+".0000"+str(i)+".wav"
    scale, sr = librosa.load(song)
    mel_spectrogram = librosa.feature.melspectrogram(scale, sr=sr, n_fft=2048, hop_length=512, n_mels=10)
    mel_spectrogram.shape
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(log_mel_spectrogram, 
                            x_axis="time",
                            y_axis="mel", 
                            sr=sr)
    plt.colorbar(format="%+2.f")
    plt.savefig(parent_dir+"mel_spectograms/"+genre+"/"+genre+".0000"+str(i)+".png")

  for i in range(10,100):  
    song = parent_dir+"genres_original/"+genre+"/"+genre+".000"+str(i)+".wav"
    scale, sr = librosa.load(song)
    mel_spectrogram = librosa.feature.melspectrogram(scale, sr=sr, n_fft=2048, hop_length=512, n_mels=10)
    mel_spectrogram.shape
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(log_mel_spectrogram, 
                            x_axis="time",
                            y_axis="mel", 
                            sr=sr)
    plt.colorbar(format="%+2.f")
    plt.savefig(parent_dir+"mel_spectograms/"+genre+"/"+genre+".000"+str(i)+".png")


