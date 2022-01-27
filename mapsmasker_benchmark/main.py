import time
import warnings

warnings.filterwarnings(action='ignore')

from pathlib import Path
from nilearn.maskers import NiftiMapsMasker, NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_difumo, fetch_adhd, fetch_development_fmri
from nilearn.image import index_img, binarize_img
import numpy as np
import pandas as pd
from time import time


DIFUMO_DIM = 64


def timer_func(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


if __name__ == '__main__':

    dataset = fetch_adhd(1, data_dir=Path(__file__).parents[1] / 'data')
    atals = fetch_atlas_difumo(dimension=DIFUMO_DIM,
                               resolution_mm=3,
                               data_dir=Path(__file__).parents[1] / 'data')
    # high_dim_difumo = fetch_atlas_difumo(dimension=256, resolution_mm=3, data_dir=Path(__file__).parents[1] / 'data')

    @timer_func
    def mask_adhd_difumo(strategy, maps):
        masker = NiftiMapsMasker(maps, strategy=strategy, detrend=True)
        return masker.fit_transform(dataset.func[0], confounds=dataset.confounds[0])


    for strategy in ['ridge', 'ols']:
        print('===============================')
        print(strategy)
        print(f"Time to extract difumo {DIFUMO_DIM} dimensions:")
        mask_adhd_difumo(strategy, atals.maps)
        print("")

    r2_collector = []
    for i in range(DIFUMO_DIM):
        cur_dim = index_img(atals.maps, i)
        cur_dim_bin = binarize_img(cur_dim)
        print(f"...get R2 for {i+1}")

        masker = NiftiLabelsMasker(cur_dim_bin, strategy='mean', detrend=True)
        original = masker.fit_transform(dataset.func[0],
                                        confounds=dataset.confounds[0]).flatten()

        cur_r2 = {}
        for strategy in ['ridge', 'ols']:
            soft_masker = NiftiMapsMasker(cur_dim, strategy=strategy, detrend=True)
            soft_mask = soft_masker.fit_transform(dataset.func[0],
                                                  confounds=dataset.confounds[0]).flatten()
            r2 = np.corrcoef(original, soft_mask)[0, 1]**2
            cur_r2[strategy] = r2
        r2_collector.append(cur_r2)
    r2_collector = pd.DataFrame(r2_collector)
    print(r2_collector.corr())