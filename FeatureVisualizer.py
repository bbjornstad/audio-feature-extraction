import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt


class FeatureVisualizer:
    """
    This class implements an object that can handle the visualization of various
    features extracted from audio samples. Contains some attributes that store
    the location of extracted data and plotting defaults.

    Attributes:
    -----------
        :str feature_folder:        string representing a path to a folder
                                    containing feature matrices stored as CSV
                                    files in the standard
                                    {sample_name}_{feat_abbrev}_features.csv
                                    (default 'feature_extraction/')
        :tuple(float) figure_size:  tuple of floats that represents the desired 
                                    figure size to use for matplotlib (default 
                                    (18,8)) 
    """
    def __init__(
            self,
            feature_folder='feature_extraction/',
            figure_size=(18, 8)):
        """
        Instantiates a FeatureVisualizer object with the given parameters.

        Parameters:
        -----------
            :str feature_folder:            string representing a path to a
                                            folder containing feature matrices
                                            stored as CSV files in the standard
                                            {sample_name}_{feat_abbrev}_features
                                            .csv (default 'feature_extraction/')
            :tuple(float) figure_size:      tuple of floats representing the
                                            desired figure size to use for
                                            matplotlib (default (18,8))
        """
        self.figure_size = default_figure_size
        self.feature_folder = feature_folder

    def plot_melspec(self, sample_name):
        """
        Plots the melspectrogram of the given sample, if it exists in the
        instance's feature_folder.

        Parameters:
        -----------
            :str sample_name:               string representing the file name
                                            within the feature_folder to display
        """
        melspec_fig = plt.figure(figsize=self.figure_size)
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
        """
        Plots the chromagram of the given sample, if it exists in the instance's
        feature_folder.

        Parameters:
        -----------
            :str sample_name:               string representing the file name
                                            within the feature_folder to display
        """
        cstft_fig = plt.figure(figsize=self.figure_size)
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
        """
        Plots the spectrogram of the given sample, if it exists in the 
        instance's feature_folder.

        Parameters:
        -----------
            :str sample_name:               string representing the file name
                                            within the feature_folder to display
        """
        stft_fig = plt.figure(figsize=self.figure_size)
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
