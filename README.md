# mapsmasker_benchmark
Easy benchmark on the performance of nilearn NiftiMapsMasker

Please install nilearn through `requirements.txt`. We are using a customised version of `NiftiLabelMasker`

Dataset used: ADHD200

Two benchmarking metrics:
1. Run time of time series extraction from the 256 dimensions difumo atlas with confound regression
2. R^2 of raw time series and masker extracted from PCC (taken from difumo 64 dimensions map 4)
   The raw time series was the average signal extracted using a binarised mask of difumo 64 dimensions map 4.
