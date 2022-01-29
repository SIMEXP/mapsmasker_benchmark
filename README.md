# mapsmasker_benchmark
Easy benchmark on the performance of nilearn `NiftiMapsMasker`

⚠️ __Here we are only using binder for showing the plots. The notebook is not executable on binder hub.__ ⚠️

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SIMEXP/mapsmasker_benchmark/HEAD?urlpath=https%3A%2F%2Fgithub.com%2FSIMEXP%2Fmapsmasker_benchmark%2Fblob%2Fmain%2Fnotebooks%2Fplots.ipynb)

Please install nilearn through `requirements.txt`. We are using a customised version of `NiftiMapsMasker`

We added an option to do a simple linear fit:
https://github.com/htwangtw/nilearn/blob/ef98392f74bd42d12f4382dcca67725880943133/nilearn/regions/signal_extraction.py#L298-L305

Dataset used: ADHD200

Two benchmarking metrics:
1. Run time of time series extraction from difumo atlas with confound regression
2. Quality of compression measure as R2

