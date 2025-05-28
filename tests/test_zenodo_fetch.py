import os
import shutil

from miblab import zenodo_fetch



def test_zenodo_fetch():
    
    tmp_dir = os.path.join(os.getcwd(), 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    testdata = 'test_data_post_contrast_dixon.zip'
    testdatadoi = '15489381'

    # Download ZIP file to temp directory
    zenodo_fetch(testdata, tmp_dir, testdatadoi)
    zenodo_fetch(testdata, tmp_dir, testdatadoi)
    zenodo_fetch(testdata, tmp_dir, testdatadoi, extract=True)
    zenodo_fetch(testdata, tmp_dir, testdatadoi, extract=True)

    shutil.rmtree(tmp_dir)

    zenodo_fetch(testdata, tmp_dir, testdatadoi, extract=True)

    shutil.rmtree(tmp_dir)


if __name__=='__main__':
    test_zenodo_fetch()

    print('zenodo_fetch passed all tests..')
