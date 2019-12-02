# Python Audio Feature Extraction
This repository holds a library of implementations of a few separate utilities to be used for the extraction and processing of features from audio files. The underlying extraction library is `librosa`, which offers the ability to extract a variety of spectral features as well as a few other miscellaneous features.

## Project Goals
- To learn more about feature extraction from audio files
- To standardize parameters for use during extraction on large amounts of audio samples
- To allow for a relatively easy interface to select and extract subsets of parameters from subsets of samples
- To provide Pandas wrappings of extraction results to hold important metadata and information about the extraction process

## Current Implementations
- `AudioFeatureExtractor`: this class defines an object that can be used to standardize a set of parameters to be used during feature extraction. It provides wrapper methods to `librosa` functions and can handle preprocessing steps such as preemphasis filtering and hard low and high cutoffs to facilitate data cleaning.
- `BatchExtractor`: this class defines an object that holds information about a batch of audio samples for which feature extraction should be performed. It implements methods that handle batch extraction using a set of standardized settings and easy selection of desired features.
- `FeatureVisualizer`: this class defines an object that can handle the visualization of features through `Matplotlib`.
----

## Usage Documentation
### AudioFeatureExtractor
Each class can be imported from the `afe` module. In other words, to import the AudioFeatureExtractor object, simply put `from afe import AudioFeatureExtractor` along with the rest of your needed imports. Then, you can instantiate an AudioFeatureExtractor object and put it to work. This object takes as parameters at instantiation:
- The desired sample rate in Hz to use for loading and analysis of audio (default 22050)
- The desired number of samples to use as the window length for framed computations and feature extractions. This number should be set to an integer power of 2 to optimize the Fourier engine (default 1024)
- The desired ratio of a window length to hop during framed computations -- i.e. an overlapping factor, setting this to 4 implies that the frame jumps 1/4 of a window length during framed computations (default 4).

An `AudioFeatureExtractor`:
- has the capability of retrieving audio from a string file path at the instance's sample rate and loading it as a Numpy array
- can detect onsets, perform preemphasis filtering, and bandpass filtering for noise removal in the low and high regions
- can extract a feature using the instance's standardized framing attributes, either from a Numpy array of audio samples or from a preprocessed STFT/CQT (such as with bandpass noise removal)

Currently the following feature extraction methods are implemented (all feature extraction methods begin with the prefix `extract_`:
- `extract_stft`: extracts a short time Fourier transform
- `extract_cqt`: extracts a constant-Q transform
- `extract_chroma_stft`: extracts a chromagram
- `extract_chroma_cqt`: extracts a chromagram of a CQT
- `extract_chroma_cens`: extracts an energy normalized variant chromagram
- `extract_melspectrogram`: extracts a Mel-windowed spectrogram
- `extract_mfcc`: extracts the Mel-frequency cepstral coefficients
- `extract_rms`: extracts the framed root-mean-square
- `extract_spectral_centroid`: extracts the spectral centroid
- `extract_spectral_bandwidth`: extracts the spectral bandwidth
- `extract_spectral_contrast`: extracts the spectral contrast
- `extract_spectral_flatness`: extracts the spectral flatness
- `extract_spectral_rolloff`: extracts the spectral rolloff
- `extract_zero_crossing_rate`: extracts the framed zero crossing rate
- `extract_tonnetz`: extracts the tonnetz (tonal centroid)
- `extract_poly_features`: extracts polynomial combinations of features from a given feature matrix or audio

Ultimately I would like to add these functionalities to the `AudioFeatureExtractor`:
- Tempo related feature extraction methods
- Feature manipulation tools offered by librosa
- Feature inversion tools to translate back from the feature space to the auditory space, to hopefully facilitate some interesting generative projects later on
- Perhaps incorporating some other utilities could be helpful for later projects or methods of feature engineering.

### BatchExtractor
The `BatchExtractor` is somewhat of an extension of the `AudioFeatureExtractor` to handle standardized extraction of a batch of samples. In order to use the `BatchExtractor` object, we must import it in the same way: `from afe import BatchExtractor`. Then we can instantiate a `BatchExtractor` object. This object accepts at instantiation the following parameters:
- The same three parameters as accepted during instantiation of an `AudioFeatureExtractor` object
- A string path of a folder containing audio samples
- Either a Pandas dataframe or a string path to a CSV file that can be read as a dataframe which has metadata information about samples in the specified folder of audio. Details about the formatting of this index dataframe are discussed more in-depth below
- The number of Mel-frequency cepstral coefficients to compute (default 12)
- A boolean flag indicating whether or not preemphasis filtering should be applied to audio before computation of features (default False)
- A float between 0 and 1 indicating the desired preemphasis filter coefficient if this option is desired and appropriately set using the Boolean flag (default 0.97)
- A boolean flag indicating whether or not hard-bandpass filtering of noise in the higher and lower frequencies should be applied (default False)
- If hard-bandpass noise filtering should be applied and the flag has been appropriately set, then integer values can be set for the upper and lower limits of this noise filtering (default None)
- A boolean flag indicating whether or not the start of the audio samples should be trimmed to the first computed onset (default False)

The `BatchExtractor`, as specified, requires that an index dataframe be specified, either as a string pointing to a CSV file or by passing in the dataframe itself. In particular, this dataframe needs a column named `file_name` which indicates for each audio sample to be analyzed the file path as a string and relative to the `BatchExtractor`'s stored audio folder path. Other columns present in the index dataframe could depend on the context of the project.

Currently, the `BatchExtractor` can be used to perform the following tasks:
- There are methods available to set any of the preprocessing options:
    - `set_bp_filter`: sets the bandpass noise filtering flag and parameters
    - `set_preemphasis`: sets the preemphasis flag and parameters
    - `set_trim`: sets the flag for trimming to first onset
- There are methods to extract and merge features from the batch of samples stored in the instance's index of the folder of audio. Each of these methods accepts a string indicating a `results_folder` in which to either save the extracted feature matrices as CSV files or look for the saved extraction results as CSV files (in the case of merging). Additional options for each method are further detailed below.
    - `batch_extract_feature`: accepts a string abbreviation of a single extraction method to apply to the entire batch of audio in the index. Saves the results to the given `results_folder`.
    - `batch_extract_features`: accepts a list of string abbreviations of extraction methods to apply to the entire batch of audio in the index. Saves the results to the given `results_folder`.
    - `merge_features`: accepts a list of string abbreviations of features to merge into a single dataframe. In other words, for each sample in the audio index, each of the feature matrices specified in the list of abbreviations will be loaded from the `results_folder` and merged, then saved as a new dataframe.
    - `batch_extract_and_merge`: Performs a batch extraction then a merging of all features given in the list of string abbreviations for all audio samples in the index.
    - `merge_and_flatten_features`: The only of these methods to have a non-null return, this method loads for each sample in the audio index the feature matrices for the given list of extraction abbreviations, flattens them into a single row (creating many many columns), then concatenates all these rows into an appropriately padded dataframe containing feature information for all samples in the index.

I would eventually like to add the following to the `BatchExtractor` implementation


### FeatureVisualizer
The `FeatureVisualizer` class defines an object which could be helpful for the purposes of debugging or identifying import sonic features. In general, computations such as the STFT/spectrogram are helpful because they can give us a very good visual description of what is happening in audio, even though their quantities are also helpful by making more specific quantifications for the purposes of analysis. The `FeatureVisualizer` class is my attempt at creating an interface for the visualization of features extracted using the `BatchExtractor` (or similarly `AudioFeatureExtractor`) object. At instantiation, this object accepts the following parameters:
- A string path indicating a folder containing extracted features with the following naming convention: `'{sample_name}_{feature_abbreviation}_features.csv'`
- A default figure size to use when plotting (default (18, 8)).

This is currently the class for which the least implementation has been written. Currently the object is capable of visualizing by simply calling the name of the sample to be loaded from the folder of extracted features.
- STFT/spectrograms
- Mel-windowed spectrograms
- Chromagrams

There are many more feature visualization methods that I would like to implement (CQTs, spectral bandwidth and related features, tempograms eventually). 
