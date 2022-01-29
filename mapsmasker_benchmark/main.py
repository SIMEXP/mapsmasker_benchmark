import time
import warnings

warnings.filterwarnings(action='ignore')

from pathlib import Path
import numpy as np

from nilearn.maskers import NiftiMapsMasker, NiftiMasker
from nilearn.datasets import fetch_atlas_difumo, fetch_adhd
from nilearn.image import load_img, mean_img, binarize_img

from time import time


DIFUMO_DIM = [64, 128, 256, 512, 1024]


def timer_func(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


if __name__ == '__main__':
    results = Path(__file__).parents[1] / 'results'
    dataset = fetch_adhd(6, data_dir=Path(__file__).parents[1] / 'data')
    func = load_img(dataset.func[0])
    conf = dataset.confounds[0]
    mask = binarize_img(mean_img(func))
    # standardise original data
    preprocessor = NiftiMasker(standardize=True, detrend=True,
                               low_pass=0.1, high_pass=0.01, t_r=2.,
                               smoothing_fwhm=5,
                               memory=str(Path(__file__).parents[1] / 'nilearn_cache'),
                               memory_level=1, verbose=0)
    data = preprocessor.fit_transform(func, conf)
    data_nii = preprocessor.inverse_transform(data)

    @timer_func
    def mask_adhd_difumo(strategy, maps):
        masker = NiftiMapsMasker(maps, strategy=strategy)
        timeseries = masker.fit_transform(data_nii)
        return timeseries, masker

    r2_scores = {}
    for dim in DIFUMO_DIM:
        print(f'================={dim}=================')
        atlas = fetch_atlas_difumo(dimension=dim,
                                   resolution_mm=3,
                                   data_dir=Path(__file__).parents[1] / 'data')
        r2_scores[dim] = {}
        for strategy in ['ols', 'ridge',]:
            print(strategy)
            if strategy != 'dypac':
                print(f"Time to extract difumo {dim} dimensions:")
                timeseries, masker = mask_adhd_difumo(strategy, atlas.maps)
                compressed_raw = masker.inverse_transform(timeseries)
                standard_masker = NiftiMasker(mask, standardize=True)  # the scale is weird for ols
                compressed = standard_masker.fit_transform(compressed_raw)
                r2 = 1 - (np.var(data - compressed, axis=0) / np.var(data, axis=0))
                r2_map = standard_masker.inverse_transform(r2)
            r2_scores[dim][strategy] = {'r2': r2_map,
                                        'compressed_z': compressed,
                                        'compressed_raw':compressed_raw}
    print(f'=====================================')
