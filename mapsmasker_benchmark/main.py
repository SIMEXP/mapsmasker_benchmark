import time
import warnings

warnings.filterwarnings(action='ignore')

from pathlib import Path
from nilearn.maskers import NiftiMapsMasker, NiftiMasker
from nilearn.datasets import fetch_atlas_difumo, fetch_adhd
from nilearn.image import load_img, math_img, mean_img, binarize_img
from nilearn.plotting import view_img
import numpy as np
import nibabel as nb
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
    dataset = fetch_adhd(1, data_dir=Path(__file__).parents[1] / 'data')
    func = load_img(dataset.func[0])
    conf = dataset.confounds[0]
    mask = binarize_img(mean_img(func))
    # standardise original data
    preprocessor = NiftiMasker(standardize=True, detrend=True)
    data = preprocessor.inverse_transform(preprocessor.fit_transform(func, conf))


    @timer_func
    def mask_adhd_difumo(strategy, maps):
        masker = NiftiMapsMasker(maps, strategy=strategy,
                                 standardize=True, detrend=True,
                                 resampling_target='data')
        timeseries = masker.fit_transform(func, conf)
        return timeseries, masker


    for dim in DIFUMO_DIM:
        print(f'================={dim}=================')

        atlas = fetch_atlas_difumo(dimension=dim,
                                resolution_mm=3,
                                data_dir=Path(__file__).parents[1] / 'data')
        r2_scores = {}
        for strategy in ['ridge', 'ols']:
            print(strategy)
            print(f"Time to extract difumo {dim} dimensions:")
            timeseries, masker = mask_adhd_difumo(strategy, atlas.maps)
            compressed = masker.inverse_transform(timeseries)
            r2 = math_img("1 - np.var(data - compressed, axis=-1)", data=data, compressed=compressed)
            r2 = math_img("mask * img", mask=mask, img=r2)
            r2_scores[strategy] = [r2, compressed]
            view = view_img(r2)
            view.save_as_html(str(results / f'extraction-{strategy}_difumo-{dim}.html'))
    print(f'=====================================')
