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