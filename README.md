# Python Audio Feature Extraction
This repository holds a library of implementations of a few separate utilities to be used for the extraction and processing of features from audio files. The underlying extraction library is Librosa, which offers the ability to extract a variety of spectral features as well as a few other miscellaneous features.

## Project Goals
- To learn more about feature extraction from audio files
- To standardize parameters for use during extraction on large amounts of audio samples
- To allow for a relatively easy interface to select and extract subsets of parameters from subsets of samples
- To provide Pandas wrappings of extraction results to hold important metadata and information about the extraction process

## Current Implementations
- AudioFeatureExtractor: this class defines an object that can be used to standardize a set of parameters to be used during feature extraction. It provides wrapper methods to librosa functions and can handle preprocessing steps such as preemphasis filtering and hard low and high cutoffs to facilitate data cleaning.
- BatchExtractor: this class defines an object that holds information about a batch of audio samples for which feature extraction should be performed. It implements methods that handle batch extraction using a set of standardized settings and easy selection of desired features.
- FeatureVisualizer: this class defines an object that can handle the visualization of features through matplotlib.
----

## Usage Documentation
### AudioFeatureExtractor
Each class can be imported from the `afe` module. In other words, to import the AudioFeatureExtractor object, simply put `from afe import AudioFeatureExtractor` along with the rest of your needed imports. Then, you can instantiate an AudioFeatureExtractor object and put it to work. This object takes as parameters at instantiation:
- The desired sample rate in Hz to use for loading and analysis of audio (default 22050)
- The desired number of samples to use as the window length for framed computations and feature extractions. This number should be set to an integer power of 2 to optimize the Fourier engine (default 1024)
- The desired ratio of a window length to hop during framed computations -- i.e. an overlapping factor, setting this to 4 implies that the frame jumps 1/4 of a window length during framed computations (default 4).

An AudioFeatureExtractor:
- has the capability of retrieving audio from a string file path at the instance's sample rate and loading it as a Numpy array
- can detect onsets, perform preemphasis filtering, and bandpass filtering for noise removal in the low and high regions
- can extract a feature using the instance's standardized framing attributes, either from a Numpy array of audio samples or from a preprocessed STFT/CQT (such as with bandpass noise removal)

Currently the following feature extraction methods are implemented (all feature extraction methods begin with the prefix `extract_`:
- extract_stft: extracts a short time Fourier transform
- extract_cqt: extracts a constant-Q transform

Ultimately I would like to add these features to the AudioFeatureExtractor:
- Feature manipulation tools offered by librosa
- Feature inversion tools to translate back from the feature space to the auditory space, to hopefully facilitate some interesting generative projects later on
- Perhaps incorporating some other utilities could be helpful for later projects or methods of feature engineering.

### BatchExtractor
In order to use the BatchExtractor object, we must import it in the same way" `from afe import BatchExtractor`. 