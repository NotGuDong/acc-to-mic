import os

import librosa
import numpy as np
import pandas as pd
from scipy import signal


def wavToMel(file, savePath):
    sample, sr = librosa.load(file, sr=16000)
    mel_spec = librosa.feature.melspectrogram(y=sample, sr=sr, n_fft=512, hop_length=256)
    np.save(savePath, mel_spec)


def getStft(data, sr, n_fft=128, hop_length=32):
    _, _, Zxx = signal.stft(data, fs=sr, window='hann', noverlap=n_fft - hop_length)
    return np.abs(Zxx)


def csvToSpec(csvFile, savePath):
    df = pd.read_csv(csvFile)
    zData = df.iloc[:, 2].values
    sr = 417
    Zxx = getStft(zData, sr)
    np.save(savePath, Zxx)


if __name__ == '__main__':
    wavDir = "F:\\python\\Dataset\\LibriSpeech\\train-clean\\train-clean-100\\"
    csvDir = "F:\\python\\Dataset\\LibriSpeech\\patch000_LibriSpeech\\train-clean\\train-clean-100\\"
    wavToMelOutDir = "F:\\python\\Dataset\\LibriSpeech\\wav_to_mel_npy\\"
    csvToSpecOutDir = "F:\\python\\Dataset\\LibriSpeech\\csv_to_spec_npy\\"

    os.makedirs(wavToMelOutDir, exist_ok=True)
    os.makedirs(csvToSpecOutDir, exist_ok=True)

    wavPatch1 = os.listdir(wavDir)
    for patch1 in wavPatch1:
        if not os.path.exists(os.path.join(wavToMelOutDir, patch1)):
            os.mkdir(os.path.join(wavToMelOutDir, patch1))
        wavPatch2 = os.listdir(os.path.join(wavDir, patch1))
        for patch2 in wavPatch2:
            if not os.path.exists(os.path.join(wavToMelOutDir, patch1, patch2)):
                os.mkdir(os.path.join(wavToMelOutDir, patch1, patch2))
            fileList = os.listdir(os.path.join(wavDir, patch1, patch2))
            for fileName in fileList:
                if not fileName.endswith(".flac"):
                    continue
                wavToMel(os.path.join(wavDir, patch1, patch2, fileName),
                         os.path.join(wavToMelOutDir, patch1, patch2,
                                      fileName.replace(".flac", ".npy")))

    csvPatch1 = os.listdir(csvDir)
    for patch1 in csvPatch1:
        if not os.path.exists(os.path.join(csvToSpecOutDir, patch1)):
            os.mkdir(os.path.join(csvToSpecOutDir, patch1))
        csvPatch2 = os.listdir(os.path.join(csvDir, patch1))
        for patch2 in csvPatch2:
            if not os.path.exists(os.path.join(csvToSpecOutDir, patch1, patch2)):
                os.mkdir(os.path.join(csvToSpecOutDir, patch1, patch2))
            fileList = os.listdir(os.path.join(csvDir, patch1, patch2))
            for fileName in fileList:
                if not fileName.endswith(".csv"):
                    continue
                csvToSpec(os.path.join(csvDir, patch1, patch2, fileName),
                          os.path.join(csvToSpecOutDir, patch1, patch2,
                                       fileName.replace(".csv", ".npy")))
