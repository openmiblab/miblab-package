import os
import shutil

import dbdicom as db
import numpy as np

from miblab import kidney_pc_dixon
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
    arrays = (
        db.pixel_data(study + ['Dixon_post_contrast_out_phase']),
        db.pixel_data(study + ['Dixon_post_contrast_in_phase']),
        db.pixel_data(study + ['Dixon_post_contrast_water']),
        db.pixel_data(study + ['Dixon_post_contrast_fat']),
    )
    array = np.stack(arrays, axis=-1)

    mask = kidney_pc_dixon(array, model='unetr', verbose=True)
    assert np.sum(mask['leftkidney']) == 62284

    mask = kidney_pc_dixon(array, model='nnunet', verbose=True)
    assert np.sum(mask['leftkidney']) == 79285

    shutil.rmtree(tmp_dir)


if __name__=='__main__':
    test_kidney_pc_dixon()
    print('kidney_pc_dixon passed all tests..')
