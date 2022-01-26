import time
import warnings

warnings.filterwarnings(action='ignore')

from pathlib import Path
from nilearn.maskers import NiftiMapsMasker, NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_difumo, fetch_adhd
from nilearn.image import index_img, binarize_img
import numpy as np
from time import time


def timer_func(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


if __name__ == '__main__':

    adhd = fetch_adhd(1, data_dir=Path(__file__).parents[1] / 'data')
    atals = fetch_atlas_difumo(dimension=64, resolution_mm=3, data_dir=Path(__file__).parents[1] / 'data')
    high_dim_difumo = fetch_atlas_difumo(dimension=256, resolution_mm=3, data_dir=Path(__file__).parents[1] / 'data')

    difumo_pcc = index_img(atals.maps, 3)
    bin_pcc = binarize_img(difumo_pcc)


    @timer_func
    def mask_adhd_difumo(strategy, maps):
        masker = NiftiMapsMasker(maps, strategy=strategy, detrend=True)
        return masker.fit_transform(adhd.func[0], confounds=adhd.confounds[0])

    # binarise difumo and get original time series
    masker = NiftiLabelsMasker(bin_pcc, strategy='mean', detrend=True)
    original = masker.fit_transform(adhd.func[0], confounds=adhd.confounds[0]).flatten()

    for strategy in ['ridge', 'ols']:
        print('===============================')
        print(strategy)
        print("Time to extract difumo 256 dimensions:")
        mask_adhd_difumo(strategy, high_dim_difumo.maps)
        print("")
        print("...get r-squred from pcc")
        ols = mask_adhd_difumo(strategy, difumo_pcc)
        ols = ols.flatten()
        r2 = np.corrcoef(original, ols)[0, 1]**2
        print(f'PCC r-squred: {r2}')
