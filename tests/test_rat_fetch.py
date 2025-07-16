"""
tests/test_rat_fetch.py
=======================

Integration test for :pyfunc:`miblab.rat_fetch`.

The test is only executed when Zenodo is reachable.  It is marked
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

from __future__ import annotations

import socket
from pathlib import Path
from typing import List

import pytest
import requests

from miblab import rat_fetch
from miblab.data import _have_dicom2nifti   # feature-flag published by the lib

#  Helpers                                                                    

def _zenodo_online() -> bool:
    """
    Minimal reachability probe for *zenodo.org*.

    Returns *False* on any DNS or HTTP failure so that the entire test can be
    skipped gracefully when the CI runner is offline.
    """
    try:
        socket.gethostbyname("zenodo.org")
        return requests.head("https://zenodo.org/", timeout=5).status_code == 200
    except Exception:  # noqa: BLE001 – network problems ⇒ offline
        return False


#  Pytest markers                                                             

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        not _zenodo_online(),
        reason="Zenodo unreachable; skipping rat_fetch test.",
    ),
]

#  Parameterised smoke / pipeline test                                        

@pytest.mark.parametrize(
    "dataset, unzip, convert",
    [
        pytest.param("S01", False, False, id="S01-download_only"),
        pytest.param(
            "S01",
            True,
            True,
            id="S01-unzip+convert",                     # shown only when dicom2nifti is present
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
    Exercise :pyfunc:`miblab.rat_fetch` under two configurations.

    * Any transient 502 / 503 / connection failure → ``pytest.skip``  
    * All other exceptions                         → **test failure**
    """
    download_dir = tmp_path / "downloads"

    # ---------------------------------------------------------------- download
    try:
        returned: List[str] = rat_fetch(
            dataset=dataset,
            folder=download_dir,
            unzip=unzip,
            convert=convert,
        )
    except Exception as exc:  # noqa: BLE001
        if any(code in str(exc) for code in ("502", "503", "ConnectionError")):
            pytest.skip(f"Zenodo transient error ({exc}); skipping.")
        raise

    # ---------------------------------------------------------------- asserts
    assert download_dir.exists(), "Download folder was not created"

    # 1 ZIP present
    assert list(download_dir.glob("*.zip")), "No ZIP files downloaded"

    # returned paths exist
    assert returned, "Function returned an empty list of paths"
    for p in returned:
        assert Path(p).exists(), f"Returned path {p} does not exist"

    # if we unzipped, there must be at least one DICOM
    if unzip:
        assert any(download_dir.rglob("*.dcm")), "No DICOMs found after unzip"

    # if we converted, there must be at least one NIfTI
    if convert:
        nifti_root = download_dir.parent / f"{download_dir.name}_nifti"
        nii_found = any(nifti_root.rglob("*.nii")) or any(
            nifti_root.rglob("*.nii.gz")
        )
        assert nii_found, "No NIfTI files produced"

    print(
        f"[OK] rat_fetch(dataset={dataset!r}, unzip={unzip}, convert={convert}) passed."
    )