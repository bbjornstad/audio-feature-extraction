import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt


class FeatureVisualizer:
    def __init__(
            self,
            feature_folder='feature_extraction/',
            default_figure_size=(18, 8)):
        self.default_figure_size = default_figure_size
        self.feature_folder = feature_folder

    def plot_melspec(self, sample_name):
        melspec_fig = plt.figure(figsize=self.default_figure_size)
        try:
            melspec = pd.read_csv(
                f'{self.feature_folder}{sample_name}_melspec_features.csv').T
            melspec = melspec.to_numpy()
        except FileNotFoundError:
            print('Feature matrix not found...did you remember to extract the features?')
        melspec_db = librosa.power_to_db(melspec, ref=np.max)
        librosa.display.specshow(melspec_db, x_axis='time', y_axis='mel')
        plt.title(f'Melspectrogram -- {sample_name}')
        return melspec_fig

    def plot_chromagram(self, sample_name):
        cstft_fig = plt.figure(figsize=self.default_figure_size)
        try:
            cstft = pd.read_csv(
                f'{self.feature_folder}{sample_name}_cstft_features.csv').T
            cstft = cstft.to_numpy()
        except FileNotFoundError:
            print('Feature matrix not found...did you remember to extract the features?')
        librosa.display.specshow(cstft, x_axis='time', y_axis='chroma')
        plt.title(f'Chromagram -- {sample_name}')
        return cstft_fig

    def plot_spectrogram(self, sample_name):
        stft_fig = plt.figure(figsize=self.default_figure_size)
        try:
            stft = pd.read_csv(
                f'{self.feature_folder}{sample_name}_stft_features.csv').T
            stft = stft.to_numpy()
        except FileNotFoundError:
            print('Feature matrix not found...did you remember to extract the features?')
        stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        librosa.display.specshow(stft_db, y_axis='log')
        plt.title(f'Spectrogram -- {sample_name}')
        return stft_fig
