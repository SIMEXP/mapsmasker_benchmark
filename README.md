# mapsmasker_benchmark
Easy benchmark on the performance of nilearn NiftiMapsMasker

Please install nilearn through `requirements.txt`. We are using a customised version of `NiftiLabelMasker`

We added an option to do a simple linear fit:
https://github.com/htwangtw/nilearn/blob/ef98392f74bd42d12f4382dcca67725880943133/nilearn/regions/signal_extraction.py#L298-L305

Dataset used: ADHD200

Two benchmarking metrics:
1. Run time of time series extraction from the 256 dimensions difumo atlas with confound regression
2. R^2 of raw time series and masker extracted from PCC (taken from [difumo 64 dimensions map 4](https://parietal-inria.github.io/DiFuMo/64/html/4.html))
   The raw time series was the average signal extracted using a binarised mask of [difumo 64 dimensions map 4](https://parietal-inria.github.io/DiFuMo/64/html/4.html).
