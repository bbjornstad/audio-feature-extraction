import numpy as np
import librosa


class AudioFeatureExtractor:
    """
    This class implements an extraction object for audio samples.
    Mostly this is a wrapper for the librosa module, which has low level
    implementations of feature extraction. In particular it standardizes
    a set of parameters for the feature extraction process so that all
    samples analyzed with a particular instance have consistent framing and
    features.

    Attributes:
    -----------
        :int sr:            desired sample rate of audio
        :int frame_length:  an integer power of 2 representing the length of
                            fft and other windows
        :int hop_length:    computed from the frame length, represents the 
                            length that windows jump (currently 1/4 a window)
        :int n_mfcc:        the desired number of Mel-frequency cepstral
                            coefficients to compute
        :int fmin:          the minimum frequency used to compute certain 
                            features
        :int fmax:          the maximum frequency used to compute certain
                            features
    """

    def __init__(
            self,
            sr=22050,
            frame_length=1024,
            n_mfcc=20,
            bp_filter=False,
            fmin=1024,
            fmax=8192,
            preemphasis=False,
            pre_filter_coef=0.97):
        """
        Initializes an AudioFeatureExtractor object

        Parameters:
        -----------
            :int sr:            integer for the desired sample rate
                                (default 22050)
            :int frame_length:  an integer power of two for the frame length to
                                be used in feature extraction (default 1024)
            :int n_mfcc:        the number of Mel-frequency cepstral
                                coefficients to compute (default 20)
            :bool bp_filter:    boolean identifying whether stfts should be
                                hard bandpass filtered on computation (default
                                False)
            :int fmin:          integer for the lowest frequency that will be
                                computed in certain features (default 1024)
            :int fmax:          integer for the highest frequency that will be
                                computed in certain features.
            :bool preemphasis:  a boolean indicating whether or not preemphasis
                                filter should be applied to the audio upon
                                loading with get_audio (default False)
            :float pre_filter_coef: a float between 0 and 1 that represents the
                                filter coefficient to be used in preemphasis
                                filtering (default 0.97)
        """
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = int(self.frame_length / 4)
        self.n_mfcc = n_mfcc
        self.bp_filter = bp_filter
        self.fmin = fmin
        self.fmax = fmax
        self.preemphasis = preemphasis
        self.pre_filter_coef = pre_filter_coef

        # -----
        # Utilities
        # -----
    def get_audio(self, file_path):
        """
        Gets audio as a numpy array with the object's sample rate from the 
        given string file path.
        """
        x, sr = librosa.load(file_path, sr=self.sr)
        if self.preemphasis:
            x = self.apply_preemphasis(x)
        return x

    def apply_preemphasis(self, audio):
        """
        Applies a preemphasis filter to the given audio. A preemphasis filter is
        a simple first-order differencing filter applied to audio, which can
        have the effect of emphasizing features.
        """
        pre = librosa.effects.preemphasis(audio, coef=self.pre_filter_coef)
        return pre

    def set_preemphasis(self, flag_apply, filter_coef=0.97):
        """
        Sets the preemphasis parameters for the apply_preemphasis function.

        Parameters:
        -----------
            :bool flat_apply:       boolean flag indicating whether or not
                                    preemphasis should be applied when loading
                                    audio with get_audio
            :float filter_coef:     float between 0 and 1 indicating the desired
                                    filter coeffecient to use when applying
                                    preemphasis filtering (default 0.97)
        """
        self.preemphasis = flag_apply
        self.pre_filter_coef = filter_coef

    def bp_filter_stft(self, S):
        """
        Removes sections of the given stft below and above the given min and max
        frequencies, effectively a hard bandpass filter between the given
        frequency window. Set the params with set_bp_filter.

        Parameters:
        -----------
            :ndarray S:             stft array
        """
        fft_freqs = librosa.fft_frequencies(self.sr, self.frame_length)
        # get the indices where the frequencies occur
        min_index = next(i for i, f in enumerate(fft_freqs) if f > self.fmin)
        max_index = next(i for i, f in enumerate(fft_freqs) if f > self.fmax)

        s_filt = S
        s_filt[:min_index, :] = 0
        s_filt[max_index:, :] = 0
        return s_filt

        return S[min_index:max_index, :]

    def set_bp_filter(self, flag_apply, fmin, fmax):
        self.bp_filter = flag_apply
        self.fmin = fmin
        self.fmax = fmax

        # -----
        # Feature Extraction
        # -----
    def extract_stft(self, audio):
        """
        Extracts the short term fourier transform of the given audio, a process
        in which an audio sample is chunked into frames and the associated
        frequency energy content is analyzed, thus transforming from the time
        domain to the frequency domain (in timed chunks).
        """
        stft = librosa.stft(
            audio,
            n_fft=self.frame_length,
            hop_length=self.hop_length)
        if self.bp_filter:
            stft = self.bp_filter_stft(stft)
        return stft

    def extract_chroma_stft(self, audio):
        """
        Extracts a chromagram of the given audio, like a spectrogram but binned 
        into the chromatic scale.
        """
        S = np.abs(self.extract_stft(audio))
        chroma_stft = librosa.feature.chroma_stft(
            S=S,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        return chroma_stft

    def extract_chroma_cqt(self, audio):
        """
        Extracts a constant-Q chromagram of the given audio.
        """
        chroma_cqt = librosa.feature.chroma_cqt(
            audio,
            sr=self.sr,
            hop_length=self.hop_length)
        return chroma_cqt

    def extract_chroma_cens(self, audio):
        """
        Extracts an Energy Normalized chromagram of the given audio.
        """
        chroma_cens = librosa.feature.chroma_cens(
            audio,
            sr=self.sr,
            hop_length=self.hop_length)
        return chroma_cens

    def extract_melspectrogram(self, audio):
        """
        Extracts a Mel-windowed spectrogram of the given audio.
        """
        melspectrogram = librosa.feature.melspectrogram(
            audio,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax)
        return melspectrogram

    def extract_mfcc(self, audio):
        """
        Extracts a number of Mel-frequency cepstral coefficients from the
        given audio, where the number is controlled as an object attribute.
        """
        mfcc = librosa.feature.mfcc(
            audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax)
        return mfcc

    def extract_rms(self, audio):
        """
        Extracts the root-mean-square value for each frame of the given audio.
        """
        rms = librosa.feature.rms(
            audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        return rms

    def extract_spectral_centroid(self, audio):
        """
        Extracts the spectral centroid of the given audio.
        """
        spectral_centroid = librosa.feature.spectral_centroid(
            audio,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        return spectral_centroid

    def extract_spectral_bandwidth(self, audio):
        """
        Extracts the spectral bandwidth of the given audio.
        """
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            audio,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
        )
        return spectral_bandwidth

    def extract_spectral_contrast(self, audio):
        """
        Extracts the spectral contrast of the given audio.
        """
        spectral_contrast = librosa.feature.spectral_contrast(
            audio,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        return spectral_contrast

    def extract_spectral_flatness(self, audio):
        """
        Extracts the spectral flatness of the given audio.
        """
        spectral_flatness = librosa.feature.spectral_flatness(
            audio,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        return spectral_flatness

    def extract_spectral_rolloff(self, audio):
        """
        Extracts the spectral rolloff of the given audio.
        """
        spectral_rolloff = librosa.feature.spectral_rolloff(
            audio,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        return spectral_rolloff

    def extract_poly_features(self, audio, poly_order=3):
        """
        Extracts polynomial features from the spectrogram of the given audio,
        using the optionally specified poly_order parameter to control the
        degree (default 3).
        """
        poly_features = librosa.feature.poly_features(
            audio,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
            order=poly_order
        )
        return poly_features

    def extract_tonnetz(self, audio):
        """
        Extracts the tonnetz (tonal centroid features) from the given audio.
        """
        tonnetz = librosa.feature.tonnetz(
            audio,
            sr=self.sr
        )
        return tonnetz

    def extract_zero_crossing_rate(self, audio):
        """
        Extracts the zero crossing rate from the given audio.
        """
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length)
        return zero_crossing_rate

    def extract_tempogram(self, audio):
        """
        Extracts the tempogram from the given audio.

        ---Currently Unimplemented---
        """
        pass

    def extract_fourier_tempogram(self, audio):
        """
        Extracts the fourier tempogram from the given audio.

        ---Currently Unimplemented---
        """
        pass

        # -----
        # Feature Manipulation
        # -----
    def feature_delta(self, audio):
        pass

    def feature_stack_delta(self, audio):
        pass

    # -----
    # Feature Inversion
    # -----
    def inverse_mel_to_stft(self, melspectrogram):
        pass

    def inverse_mel_to_audio(self, melspectrogram):
        pass

    def inverse_mfcc_to_mel(self, mfcc):
        pass

    def inverse_mfcc_to_audio(self, mfcc):
        pass
