import os
import shutil

import dbdicom as db
import numpy as np
import vreg

from miblab import kidney_pc_dixon, kidney_pc_dixon_unetr
from miblab import zenodo_fetch



def test_kidney_pc_dixon():
    
    tmp_dir = os.path.join(os.getcwd(), 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    testdata = 'test_data_post_contrast_dixon.zip'
    testdatadoi = '15489381'

    # Download ZIP file to temp directory
    folder = zenodo_fetch(testdata, tmp_dir, testdatadoi, extract=True)

    # Read DICOM data
    study = db.studies(folder)[0]
    series = [
        'Dixon_post_contrast_out_phase', 
        'Dixon_post_contrast_in_phase', 
        'Dixon_post_contrast_water',
        'Dixon_post_contrast_fat',
    ]
    arrays = [db.pixel_data(study + [s]) for s in series]
    array = np.stack(arrays, axis=-1)
    vol = db.volume(study + [series[0]])

    mask = kidney_pc_dixon_unetr(vreg.volume(array, vol.affine), verbose=True)
    assert np.sum(mask.values==1) == 73871

    mask = kidney_pc_dixon(array, verbose=True)
    assert np.sum(mask==1) == 79285

    shutil.rmtree(tmp_dir)


if __name__=='__main__':
    test_kidney_pc_dixon()
    print('kidney_pc_dixon passed all tests..')
