import os
import shutil
import pytest
import requests
from miblab import osf_fetch

PROJECT_ID = "u7a6f"                  # public OSF project ID
DATASET = "Challenge_Guideline"       # dataset (folder) inside the project
LOCAL_DIR = "test_download"           # where files will be downloaded

# Check if OSF is reachable
OSF_PING_URL = f"https://api.osf.io/v2/nodes/{PROJECT_ID}"
try:
    OSF_UP = requests.head(OSF_PING_URL, timeout=5).status_code == 200
except requests.exceptions.RequestException:
    OSF_UP = False

@pytest.mark.skipif(not OSF_UP, reason="OSF unreachable or returned non-200; skipping download test.")
def test_osf_fetch():
    """Download a public dataset and verify the files appear locally."""
    # Remove leftover test folder (if any)
    if os.path.exists(LOCAL_DIR):
        shutil.rmtree(LOCAL_DIR)

    # Run the download
    try:
        osf_fetch(
            dataset=DATASET,
            folder=LOCAL_DIR,
            project=PROJECT_ID,
            extract=True,
            verbose=True,
        )
    except Exception as e:
        assert False, f"osf_fetch raised an exception: {e}"

    # Check that something was saved
    assert os.path.exists(LOCAL_DIR), "Folder not created"
    assert any(os.scandir(LOCAL_DIR)), "No files downloaded"

    # Remove the download folder
    shutil.rmtree(LOCAL_DIR)

if __name__ == "__main__":
    test_osf_fetch()