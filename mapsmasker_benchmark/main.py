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


if __name__ == '__main__':
    results = Path(__file__).parents[1] / 'results'
    dataset = fetch_adhd(6, data_dir=Path(__file__).parents[1] / 'data')
    func = load_img(dataset.func[0])
    conf = dataset.confounds[0]
    mask = binarize_img(mean_img(func))
    # standardise original data for consistent comparison
    preprocessor = NiftiMasker(standardize=True, detrend=True,
                               memory=str(Path(__file__).parents[1] / 'nilearn_cache'),
                               memory_level=1, verbose=0)
    data = preprocessor.fit_transform(func, conf)
    data_nii = preprocessor.inverse_transform(data)

    r2_scores = {}
    for dim in DIFUMO_DIM:
        print(f'================={dim}=================')
        atlas = fetch_atlas_difumo(dimension=dim,
                                   resolution_mm=3,
                                   data_dir=Path(__file__).parents[1] / 'data')
        r2_scores[dim] = {}
        for strategy in ['original', 'ols']:
            print(strategy)
            masker = NiftiMapsMasker(
                atlas.maps,
                mask_img=preprocessor.mask_img_,
                strategy=strategy).fit()

            t1 = time()
            timeseries = masker.transform(data_nii)
            t2 = time()
            print(f'NiftiMapsMasker transform data:{(t2-t1):.4f}s')

            compressed_nii = masker.inverse_transform(timeseries)

            # post processing
            standard_masker = NiftiMasker(masker._resampled_mask_img_)
            compressed = standard_masker.fit_transform(compressed_nii)

            # r2 cacluation
            r2 = 1 - (np.var(data - compressed, axis=0) / np.var(data, axis=0))
            r2_map = standard_masker.inverse_transform(r2)
            print(f"Average r2: {np.mean(r2)}")
            r2_scores[dim][strategy] = {'r2': r2_map,
                                        'compressed_raw':compressed_nii}
    print('=====================================')
