import os
import shutil

import dbdicom as db
import numpy as np
from scipy.stats import pearsonr

from miblab import kidney_dixon_fat_water
from miblab import zenodo_fetch


def test_kidney_dixon_fat_water():
    
    tmp_dir = os.path.join(os.getcwd(), 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    testdata = 'test_data_post_contrast_dixon.zip'
    testdatadoi = '15489381'

    # Download ZIP file to temp directory
    folder = zenodo_fetch(testdata, tmp_dir, testdatadoi, extract=True)

    # Read DICOM data
    study = db.studies(folder)[0]
    arrays = (
        db.pixel_data(study + ['Dixon_post_contrast_out_phase']),
        db.pixel_data(study + ['Dixon_post_contrast_in_phase']),
    )

    array = np.stack(arrays, axis=-1)

    fatwatermap = kidney_dixon_fat_water(array, verbose=True, clear_cache =True)
    
    arraystest = (
        db.pixel_data(study + ['Dixon_post_contrast_fat'])
    )
    r, _ = pearsonr(arraystest.ravel(), fatwatermap['fat'].ravel())
    assert r > 0.98942

    shutil.rmtree(tmp_dir)

if __name__=='__main__':
    test_kidney_dixon_fat_water()
    print('kidney_dixon_fat_water passed all tests..')
