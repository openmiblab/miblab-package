import os
import pytest
import requests
from miblab import osf_upload

# Public OSF project used for the test
PROJECT = "un5ct"

# Quick health-check URL to see if OSF is online
PING_URL = f"https://api.osf.io/v2/nodes/{PROJECT}"
try:
    OSF_UP = requests.head(PING_URL, timeout=5).status_code == 200
except requests.exceptions.RequestException:
    OSF_UP = False

# Skip when no token is available
@pytest.mark.skipif(
    "OSF_TOKEN" not in os.environ,
    reason="OSF_TOKEN not set; skipping upload test."
)
# Skip when OSF API is down
@pytest.mark.skipif(
    not OSF_UP,
    reason="OSF unreachable or returned non-200; skipping test."
)
def test_osf_upload_file():
    """Create a small text file and upload it to OSF."""
    test_filename = "test_upload.txt"

    # Create the dummy file
    with open(test_filename, "w") as f:
        f.write("upload test")

    # Parameters expected by osf_upload
    folder = test_filename
    dataset = f"Testing/{test_filename}"
    token = os.environ["OSF_TOKEN"]

    try:
        osf_upload(
            folder=folder,
            dataset=dataset,
            project=PROJECT,
            token=token,
            verbose=True,
            overwrite=True,
        )
    except Exception as e:
        # Treat transient 5xx or connection errors as a skip
        if any(code in str(e) for code in ("502", "503", "ConnectionError")):
            pytest.skip(f"OSF temporarily unavailable ({e}); skipping test.")
        raise
    finally:
        # Remove the local dummy file
        if os.path.exists(test_filename):
            os.remove(test_filename)

# Allow running this file directly: prints guidance if OSF_TOKEN is missing
if __name__ == "__main__":
    if "OSF_TOKEN" not in os.environ:
        print(
            "OSF_TOKEN not set. Export your token first, e.g.\n"
            "  export OSF_TOKEN=your_personal_token\n"
            "or run the suite with pytest, which will skip the test when the token is absent."
        )
        exit(0)
    test_osf_upload_file()