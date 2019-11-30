import numpy as np
import pandas as pd
import librosa
from AudioFeatureExtractor import AudioFeatureExtractor
import inspect


class BatchExtractor:
    """
    This class implements an object that can handle batch extraction of a set
    of audio samples. In particular, at instantiation a desired folder of audio
    and associated metadata index can be specified. By default, a BatchExtractor
    will look for audio files in the `raw_data` folder, and the index file is
    specified as a CSV file and defaults to `bird_vocalization_index.csv` Note
    that the index can also be specified as an already instantiated dataframe.

    Attributes:
    -----------
        :int sr:                    desired sample rate for extraction processes
                                    as an integer
        :int frame_length:          integer 2^n representing desired windowing 
                                    length for feature extraction processes
        :int n_mfcc:                integer for the desired number of 
                                    Mel-windowed cepstral coefficients to be
                                    extracted
        :AudioFeatureExtractor afe: an AudioFeatureExtractor object that handles
                                    the actual extraction for each sample
        :dict extraction_dict:      a dictionary association between string
                                    abbreviations of features and the associated
                                    extraction methods in the afe
        :str audio_folder:          a string representing a path to a folder
                                    containing raw audio files
        :df or str audio_index:     a dataframe or path to CSV that can be read
                                    as a dataframe, representing metadata
                                    and labeling information for each audio
                                    sample in the audio_folder
        :bool preemphasis:          boolean indicating whether batchwide
                                    preemphasis filtering should be applied
        :float pre_coef:            float between 0 and 1 representing the
                                    coefficient of filtering to use in
                                    preemphasis if enabled
        :bool bp_filter:            bool indicating whether batchwide hard
                                    cutoff below and above specified frequencies
                                    should be performed
        :int fmin:                  integer representing the minimum frequency
                                    that will be computed for certain spectral
                                    based features
        :int fmax:                  integer representing the maximum frequency
                                    that will be computed for certain spectral
                                    based features
        :bool trim:                 boolean indicating whether simple trimming
                                    of the start of audio should be performed
                                    using onset detection
    """

    def __init__(
            self,
            sr=22050,
            frame_length=1024,
            hop_ratio=4,
            n_mfcc=12,
            audio_folder='raw_data/',
            audio_index='bird_vocalization_index.csv',
            preemphasis=False,
            pre_coef=0.97,
            bp_filter=False,
            fmin=None,
            fmax=None,
            trim=False):
        """
        Initializes a BatchExtractor object

        Parameters:
        -----------
            :int sr:                An integer sample rate for analysis of all 
                                    files (default 22050)
            :int frame_length:      An integer power of two representing the
                                    frame length of windows used in extraction
                                    as a number of samples (default 1024)
            :int hop_ratio:         An integer representing the desired ratio of
                                    a frame to jump during windowed computations
                                    (default 4) 
            :int n_mfcc:            An integer for the number of Mel-frequency
                                    cepstral coefficients to compute (default
                                    20)
            :str audio_folder:      A string representing a path to a folder
                                    containing audio samples.
            :df or str audio_index: a dataframe or path to CSV that can be read
                                    as a dataframe and holding a metadata index
                                    for the files in the audio_folder. Must have
                                    a file_name column identifying the file
                                    location of the MP3 relative to the
                                    audio_folder.
            :bool preemphasis:      boolean value indicating whether preemphasis
                                    filtering should be applied to audio during
                                    extraction (default False)
            :float pre_coef:        float between 0 and 1 representing the
                                    preemphasis filter coefficient to use if the
                                    setting is enabled (default 0.97)
            :bool bp_filter:        boolean indicating whether or not hard
                                    bandpass filtration should be applied to
                                    spectrograms during extraction (default
                                    False)
            :int fmin:              integer representing the minimum frequency
                                    to be used for filtration of certain
                                    spectral based features if enabled
                                    (default None)
            :int fmax:              integer representing the maximum frequency
                                    to be used for filtration fo certain
                                    spectral based features if enabled
                                    (default None)
            :bool trim:             boolean indicating whether or not to apply
                                    simple trimming of the front of the audio
                                    with onset detection (default False)
        """

        self.audio_folder = audio_folder
        self.audio_index = None

        # check type on the index parameter to see if it needs to be read in.
        if isinstance(audio_index, str):
            self.audio_index = pd.read_csv(audio_index, index_col=0)
        else:
            self.audio_index = audio_index

        self.sr = sr
        self.frame_length = frame_length
        self.hop_ratio = hop_ratio
        self.n_mfcc = n_mfcc
        self.fmin = fmin
        self.fmax = fmax
        self.trim = trim

        # intialize an underlying AudioFeatureExtractor with the given params.
        self.afe = AudioFeatureExtractor(
            self.sr, self.frame_length, self.hop_ratio)

        # contains associations between string short forms and extraction
        # methods in the object's AudioFeatureExtractor
        self.extraction_dict = {'stft': self.afe.extract_stft,
                                'cqt': self.afe.extract_cqt,
                                'mfcc': self.afe.extract_mfcc,
                                'melspec': self.afe.extract_melspectrogram,
                                'zcr': self.afe.extract_zero_crossing_rate,
                                'ccqt': self.afe.extract_chroma_cqt,
                                'cstft': self.afe.extract_chroma_stft,
                                'ccens': self.afe.extract_chroma_cens,
                                'rms': self.afe.extract_rms,
                                'centroid': self.afe.extract_spectral_centroid,
                                'bandwidth': self.afe.extract_spectral_bandwidth,
                                'contrast': self.afe.extract_spectral_contrast,
                                'flatness': self.afe.extract_spectral_flatness,
                                'rolloff': self.afe.extract_spectral_rolloff,
                                'tonnetz': self.afe.extract_tonnetz,
                                'poly': self.afe.extract_poly_features
                                }

    def set_preemphasis(self, flag_apply, filter_coef=0.97):
        """
        Sets batchwide preemphasis filter settings

        Parameters:
        -----------
            :bool flag_apply:       boolean value indicating whether or not
                                    preemphasis filtering should be applied
                                    to the audio.
            :float filter_coef:     float between 0 and 1 which is the
                                    coefficient of preemphasis filtering
                                    (default 0.97)
        """
        self.preemphasis = flag_apply
        self.pre_coef = filter_coef

    def set_bp_filter(self, flag_apply, fmin, fmax):
        """
        Sets batchwide hard bandpass filter settings.

        Parameters:
        -----------
            :bool flag_apply:       boolean value indicating whether or not
                                    bandpass filtering should be applied
            :int fmin:              integer representing the minimum frequency
                                    which should be included after filtering
            :int fmax:              integer representing the maximum frequency
                                    which should be included after filtering
        """
        self.bp_filter = flag_apply
        self.fmin = fmin
        self.fmax = fmax

    def set_trim(self, flag_apply):
        """
        Sets batchwise front of audio trimming

        Parameters:
        -----------
            :bool flag_apply:       boolean value indicating whether or not to
                                    trim the beginning of each audio sample
                                    using onset detection
        """
        self.trim = flag_apply

    def batch_extract_feature(
            self,
            extraction_method,
            results_folder='feature_extraction/'):
        """
        Extracts a single feature from all of the audio files in the
        audio_folder.

        Does not return a value, but rather saves the extracted features as
        CSV files.

        Parameters:
        -----------
            :str extraction_method: a string represnting a key in the
                                    extraction_dict attribute, in other words
                                    the defined abbreviation for the desired
                                    feature
            :str results_folder:    a string representing a file path to save
                                    the extracted features (default 
                                    `feature_extraction/`)
        """
        method = self.extraction_dict[extraction_method]
        method_args = inspect.signature(method).parameters

        for file_name in self.audio_index.file_name:
            audio = self.afe.get_audio(self.audio_folder + file_name)
            if self.preemphasis:
                audio = self.afe.apply_preemphasis(audio, self.pre_coef)
            if self.trim:
                audio = self.afe.trim_start(audio)
            if self.bp_filter:
                S = self.afe.extract_stft(audio)
                S = self.afe.bp_filter_stft(S, self.fmin, self.fmax)
                if extraction_method == 'stft':
                    feature_matrix = pd.DataFrame(S.T)
                elif 'S' in method_args:
                    feature_matrix = pd.DataFrame(method(audio=None, S=S).T)
                else:
                    feature_matrix = pd.DataFrame(
                        method(audio=audio, S=None).T)
            else:
                feature_matrix = pd.DataFrame(method(audio=audio, S=None).T)

            n_cols = len(feature_matrix.columns)
            feature_cols = [f'{extraction_method}_{i}' for i in range(n_cols)]
            feature_matrix.columns = feature_cols

            name = file_name[:-4]

            feature_matrix.to_csv(
                f'{results_folder}{name}_{extraction_method}_features.csv',
                index=False)

    def batch_extract_features(
            self,
            extraction_methods,
            results_folder='feature_extraction/'):
        """
        Extracts a list of features from all the audio samples in the
        audio_folder.

        Does not return a value, but rather saves the extracted features as
        CSV files.

        Parameters:
        -----------
            :list(str) extraction_methods:  a list of strings representing
                                            abbreviations for the desired
                                            features to extract
            :str results_folder:            a string representing a file path to 
                                            save the extracted features (default 
                                            `feature_extraction/`)
        """
        for method in extraction_methods:
            print(method)
            self.batch_extract_feature(method, results_folder=results_folder)

    def merge_features(
            self,
            features_to_merge,
            results_folder='feature_extraction/'):
        """
        Merges a list of features found as extracted CSV files for each sample
        in the audio_folder.

        Parameters:
        -----------
            :list(str) features_to_merge:   a list of strings representing
                                            abbreviations for the desired
                                            features to merge
            :str results_folder:            a string representing a file path to 
                                            find the extracted features and save
                                            the merged features (default 
                                            `feature_extraction/`)
        """
        for file_name in self.audio_index.file_name:
            name = file_name[:-4]
            sample_df = pd.DataFrame()
            for feature in features_to_merge:
                feature_df = pd.read_csv(
                    f'{results_folder}{name}_{feature}_features.csv')
                sample_df = pd.concat([sample_df, feature_df], axis=1)
            sample_df.to_csv(
                f'{results_folder}{name}_merged_features.csv', index=False)

    def batch_extract_and_merge(
            self,
            extraction_methods,
            results_folder='feature_extraction/'):
        """
        Combines the batch feature extraction method and the merging method.
        """
        self.batch_extract_features(
            extraction_methods,
            results_folder=results_folder)
        self.merge_features(extraction_methods, results_folder=results_folder)

    def merge_and_flatten_features(
            self,
            extraction_methods,
            results_folder='feature_extraction/',
            label=False):
        """
        Merges the given list of features into a flattened dataframe, where each
        row represents all of the feature data for each frame for a given
        sample, and where the rows with fewer frames have those corresponding
        columns set to 0.

        Parameters:
        -----------
            :list(str) extraction_methods:  a list containing string 
                                            abbreviations for the features which
                                            we want to merge and flatten into
                                            a single dataframe.
            :str results_folder:            a string indicating a path to a
                                            folder containing feature matrices
                                            to merge
            :bool label:                    a boolean indicating whether or not
                                            each row should be labeled with its
                                            associated target.

        Returns:
        --------
            :(n, frames*n_feats) df:        a dataframe indexed by the name of
                                            the sample (derived from file name)
                                            and containing columns of the form
                                            feat_{attr}_{frame} where attribute
                                            represents a single attribute of a
                                            given feature and frame represents
                                            the frame number.
        """
        flattened_df = pd.DataFrame()

        for method in extraction_methods:
            method_df = pd.DataFrame()
            max_frames = 0
            for file_name in self.audio_index.file_name:
                name = file_name[:-4]
                feature_matrix = pd.read_csv(
                    f'{results_folder}{name}_{method}_features.csv')

                if len(feature_matrix.index) > max_frames:
                    max_frames = len(feature_matrix.index)

                col_names = list(feature_matrix.columns)
                new_row = np.ravel(feature_matrix.to_numpy(), order='F')
                new_series = pd.Series(new_row)
                new_series.name = name
                method_df = method_df.append(new_series)

            method_df.columns = [
                f'{name}_{t}' for name in col_names for t in range(max_frames)
            ]

            flattened_df = pd.concat([flattened_df, method_df], axis=1)

        if label:
            flattened_df['label'] = self.audio_index.label
        return flattened_df.fillna(0)
