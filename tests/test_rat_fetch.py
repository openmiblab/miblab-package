"""
tests/test_rat_fetch.py
=======================

Integration test for :pyfunc:`miblab.rat_fetch`.

The test only runs when Zenodo is reachable.  It is annotated
``pytest.mark.network`` so you can exclude *all* external-network tests with::

    pytest -m "not network"

Two execution paths are covered:

===========================  =====  ======
case                         unzip  convert
---------------------------  -----  ------
*download-only*  (fast)       ❌      ❌
*full pipeline* (↳ NIfTI)     ✔️      ✔️
===========================  =====  ======

Assertions
----------
1.  Target directory is created.
2.  At least one ``*.zip`` is present after download.
3.  If *unzip* = True   → at least one ``*.dcm`` exists.
4.  If *convert* = True → at least one ``*.nii[.gz]`` exists.
"""

from __future__ import annotations          # postpone annotation evaluation

import socket                               # light-weight DNS probe
from pathlib import Path
from typing import List

import pytest
import requests

from miblab import rat_fetch                # function under test
from miblab.data import _have_dicom2nifti   # feature-flag exposed by the lib


# ── Helper ────────────────────────────────────────────────────────────────
def _zenodo_online() -> bool:
    """
    Return ``True`` when *zenodo.org* resolves **and** answers HTTP HEAD, else
    ``False`` so the entire module can be skipped gracefully on offline runners.
    """
    try:
        socket.gethostbyname("zenodo.org")                       # DNS
        return requests.head("https://zenodo.org/", timeout=5).status_code == 200
    except Exception:
        return False

# ── Pytest markers (apply to whole file) ──────────────────────────────────
pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        not _zenodo_online(),
        reason="Zenodo unreachable; skipping rat_fetch test.",
    ),
]


# ── Parameterised smoke / pipeline test ──────────────────────────────────
@pytest.mark.parametrize(
    "dataset, unzip, convert",
    [
        pytest.param("S01", False, False, id="S01-download_only"),
        pytest.param(
            "S01",
            True,
            True,
            id="S01-unzip+convert",             # shown only when dicom2nifti present
            marks=pytest.mark.skipif(
                not _have_dicom2nifti,
                reason="dicom2nifti not installed – skipping conversion test",
            ),
        ),
    ],
)
def test_rat_fetch(                      
    dataset: str | None, 
    unzip: bool,
    convert: bool,
    tmp_path: Path,
) -> None:
    """
    Exercise :pyfunc:`miblab.rat_fetch` in two configurations.
    # noqa: BLE001
    * Any transient 502 / 503 / 504 or connection failure → ``pytest.skip``  
    * All other exceptions                               → **test failure**
    """
    download_dir = tmp_path / "downloads"

    try:
        returned: List[str] = rat_fetch(
            dataset=dataset,
            folder=download_dir,
            unzip=unzip,
            convert=convert,
        )
    except Exception as exc:  
        # Treat upstream hiccups as skip, everything else bubbles up
        if any(code in str(exc) for code in ("502", "503", "504", "ConnectionError")):
            pytest.skip(f"Zenodo transient error ({exc}); skipping.")
        raise

    # ── assertions ────────────────────────────────────────────────────────
    assert download_dir.exists(), "Download folder was not created"
    assert list(download_dir.glob("*.zip")), "No ZIP files downloaded"

    assert returned, "Function returned an empty list of paths"
    for p in returned:
        assert Path(p).exists(), f"Returned path {p} does not exist"

    if unzip:
        # At least one DICOM slice should exist after extraction
        assert any(download_dir.rglob("*.dcm")), "No DICOMs found after unzip"

    if convert:
        # At least one NIfTI file should exist after conversion
        nifti_root = download_dir.parent / f"{download_dir.name}_nifti"
        nii_found = any(nifti_root.rglob("*.nii")) or any(
            nifti_root.rglob("*.nii.gz")
        )
        assert nii_found, "No NIfTI files produced"

    print(f"[OK] rat_fetch(dataset={dataset!r}, unzip={unzip}, convert={convert}) passed.")