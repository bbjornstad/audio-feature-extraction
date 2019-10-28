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
        :int hop_ratio:     integer representing the ratio of a frame to jump
                            while performing framed computations
        :int hop_length:    computed from the frame length and hop ratio,
                            represents the length that windows jump in samples
        :ndarray bin_frequencies: ndarray holding the frequency values for each
                            number of fft bins that will be produced during
                            Fourier transformations.
    """

    def __init__(
            self,
            sr=22050,
            frame_length=1024,
            hop_ratio=4):
        """
        Initializes an AudioFeatureExtractor object

        Parameters:
        -----------
            :int sr:            integer for the desired sample rate
                                (default 22050)
            :int frame_length:  an integer power of two for the frame length to
                                be used in feature extraction (default 1024)
            :int hop_ratio:     integer representing the ratio of a frame length
                                to hop during framed computations (default 4)
        """
        self.sr = sr
        self.frame_length = frame_length
        self.hop_ratio = hop_ratio
        self.hop_length = int(self.frame_length / self.hop_ratio)
        self.bin_frequencies = librosa.fft_frequencies(
            self.sr,
            n_fft=self.frame_length)

        # ---------
        # Utilities
        # ---------
    def get_audio(self, file_path):
        """
        Gets audio as a numpy array with the object's sample rate from the 
        given string file path.
        """
        x, sr = librosa.load(file_path, sr=self.sr)
        return x

    def apply_preemphasis(self, audio, coef=0.97):
        """
        Applies a preemphasis filter to the given audio. A preemphasis filter is
        a simple first-order differencing filter applied to audio, which can
        have the effect of emphasizing features.
        """
        pre = librosa.effects.preemphasis(audio, coef=coef)
        return pre

    def detect_onsets(self, audio):
        onsets = librosa.onset.onset_detect(
            audio, sr=self.sr, hop_length=self.hop_length, units='samples')
        return onsets

    def trim_start(self, audio):
        onsets = self.detect_onsets(audio)
        audio = audio[onsets[0]:]
        return audio

    def bp_filter_stft(self, S, fmin, fmax):
        """
        Removes sections of the given stft below and above the given min and max
        frequencies, effectively a hard bandpass filter between the given
        frequency window.

        Parameters:
        -----------
            :ndarray S:             stft array
            :int fmin:              frequency to cut below
            :int fmax:              frequency to cut above
        """
        # get the indices where the frequencies occur
        min_index = next(i for i, f in enumerate(
            self.bin_frequencies) if f > fmin)
        max_index = next(i for i, f in enumerate(
            self.bin_frequencies) if f > fmax)

        s_filt = S
        s_filt[:min_index, :] = 0
        s_filt[max_index:, :] = 0
        return s_filt

        # ------------------
        # Feature Extraction
        # ------------------
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
        return stft

    def extract_cqt(self, audio):
        """
        Extracts the Constant-Q spectrogram transform
        """
        cqt = librosa.cqt(
            audio,
            sr=self.sr,
            hop_length=self.hop_length)
        return cqt

    def extract_chroma_stft(self, audio=None, S=None):
        """
        Extracts a chromagram of the given audio, like a spectrogram but binned 
        into the chromatic scale.
        """
        if not isinstance(S, NoneType):
            S, phase = librosa.magphase(S, power=2)
        chroma_stft = librosa.feature.chroma_stft(
            y=audio,
            S=S,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        return chroma_stft

    def extract_chroma_cqt(self, audio=None, C=None):
        """
        Extracts a constant-Q chromagram of the given audio.
        """
        chroma_cqt = librosa.feature.chroma_cqt(
            y=audio,
            C=C,
            sr=self.sr,
            hop_length=self.hop_length)
        return chroma_cqt

    def extract_chroma_cens(self, audio=None, C=None):
        """
        Extracts an Energy Normalized chromagram of the given audio.
        """
        chroma_cens = librosa.feature.chroma_cens(
            y=audio,
            C=C,
            sr=self.sr,
            hop_length=self.hop_length)
        return chroma_cens

    def extract_melspectrogram(self, audio=None, S=None):
        """
        Extracts a Mel-windowed spectrogram of the given audio.
        """
        if not isinstance(S, NoneType):
            S, phase = librosa.magphase(S, power=2)
        melspectrogram = librosa.feature.melspectrogram(
            y=audio,
            S=S,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length)
        return melspectrogram

    def extract_mfcc(self, audio=None, S=None, n_mfcc=12):
        """
        Extracts a number of Mel-frequency cepstral coefficients from the
        given audio, where the number is controlled as an object attribute.
        """
        if not isinstance(S, NoneType):
            S = self.extract_melspectrogram(audio=None, S=S)
        mfcc = librosa.feature.mfcc(
            y=audio,
            S=S,
            sr=self.sr,
            n_mfcc=n_mfcc,
            n_fft=self.frame_length,
            hop_length=self.hop_length)
        return mfcc

    def extract_rms(self, audio=None, S=None):
        """
        Extracts the root-mean-square value for each frame of the given audio.
        """
        if not isinstance(S, NoneType):
            S, phase = librosa.magphase(S)
        rms = librosa.feature.rms(
            y=audio,
            S=S,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        return rms

    def extract_spectral_centroid(self, audio=None, S=None):
        """
        Extracts the spectral centroid of the given audio.
        """
        if not isinstance(S, NoneType):
            S, phase = librosa.magphase(S)
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio,
            S=S,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        return spectral_centroid

    def extract_spectral_bandwidth(self, audio=None, S=None):
        """
        Extracts the spectral bandwidth of the given audio.
        """
        if not isinstance(S, NoneType):
            S, phase = librosa.magphase(S)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            S=S,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
        )
        return spectral_bandwidth

    def extract_spectral_contrast(self, audio=None, S=None):
        """
        Extracts the spectral contrast of the given audio.
        """
        if not isinstance(S, NoneType):
            S, phase = librosa.magphase(S)
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio,
            S=S,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        return spectral_contrast

    def extract_spectral_flatness(self, audio=None, S=None):
        """
        Extracts the spectral flatness of the given audio.
        """
        if not isinstance(S, NoneType):
            S, phase = librosa.magphase(S)
        spectral_flatness = librosa.feature.spectral_flatness(
            y=audio,
            S=S,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        return spectral_flatness

    def extract_spectral_rolloff(self, audio=None, S=None):
        """
        Extracts the spectral rolloff of the given audio.
        """
        if not isinstance(S, NoneType):
            S, phase = librosa.magphase(S)
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            S=S,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        return spectral_rolloff

    def extract_poly_features(self, audio=None, S=None, poly_order=3):
        """
        Extracts polynomial features from the spectrogram of the given audio,
        using the optionally specified poly_order parameter to control the
        degree (default 3).
        """
        if not isinstance(S, NoneType):
            S, phase = librosa.magphase(S)
        poly_features = librosa.feature.poly_features(
            y=audio,
            S=S,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
            order=poly_order
        )
        return poly_features

    def extract_tonnetz(self, audio=None, chroma=None):
        """
        Extracts the tonnetz (tonal centroid features) from the given audio.
        """
        tonnetz = librosa.feature.tonnetz(
            y=audio,
            chroma=chroma,
            sr=self.sr
        )
        return tonnetz

    def extract_zero_crossing_rate(self, audio=None, S=None):
        """
        Extracts the zero crossing rate from the given audio.
        """
        if not isinstance(S, NoneType):
            audio = librosa.istft(
                S, hop_length=self.hop_length, win_length=self.frame_length)
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

    # --------------------
    # Feature Manipulation
    # --------------------
    def feature_delta(self, audio):
        pass

    def feature_stack_delta(self, audio):
        pass

    # -----------------
    # Feature Inversion
    # -----------------
    def inverse_mel_to_stft(self, melspectrogram):
        pass

    def inverse_mel_to_audio(self, melspectrogram):
        pass

    def inverse_mfcc_to_mel(self, mfcc):
        pass

    def inverse_mfcc_to_audio(self, mfcc):
        pass
