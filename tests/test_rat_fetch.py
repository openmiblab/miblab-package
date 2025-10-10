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

Group tests
-----------
Additionally, we include smoke tests for the four friendly group names:
- rifampicin_effect_size → S01–S04  (unzip, no convert)
- six_compound           → S05,S06,S07,S08,S09, S10,S12 (unzip + convert)
- field_strength         → S13      (unzip, no convert)
- chronic                → S11,S14,S15 (unzip + convert)
Some of these are marked ``slow`` and the convert cases are auto-skipped
if ``dicom2nifti`` is not installed.
"""

from __future__ import annotations
from typing import List
from pathlib import Path
import os
import socket
import warnings

import pytest
import requests

from miblab.data import rat_fetch
from miblab.data import _have_dicom2nifti


# ── Quiet test output (optional, harmless outside pytest) ──────────────────
# Hide tqdm progress bars
os.environ.setdefault("TQDM_DISABLE", "1")
# Silence the MONAI pkg_resources deprecation warning some envs emit
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
    module="monai.utils.module",
)


# ── Helper ────────────────────────────────────────────────────────────────
def _zenodo_online() -> bool:
    """
    Return ``True`` when *zenodo.org* resolves **and** answers HTTP HEAD, else
    ``False`` so the entire module can be skipped gracefully on offline runners.
    """
    try:
        socket.gethostbyname("zenodo.org")  # DNS
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


# ── Parameterised smoke / pipeline + group tests ──────────────────────────
@pytest.mark.parametrize(
    "dataset, unzip, convert",
    [
        # Single-study fast smoke
        pytest.param("S01", False, False, id="S01-download_only"),

        # Single-study full pipeline (only when dicom2nifti available)
        pytest.param(
            "S01",
            True,
            True,
            marks=pytest.mark.skipif(
                not _have_dicom2nifti,
                reason="dicom2nifti not installed – skipping conversion test",
            ),
            id="S01-unzip+convert",
        ),

        # ── Groups (mark as slow to allow `-m "not slow"` skips) ──────────
        pytest.param(
            "rifampicin_effect_size",
            True,
            False,
            marks=pytest.mark.slow,
            id="group-rifampicin_effect_size-unzip_only",
        ),
        pytest.param(
            "six_compound",
            True,
            True,
            marks=[
                pytest.mark.slow,
                pytest.mark.skipif(
                    not _have_dicom2nifti,
                    reason="dicom2nifti not installed – skipping conversion test",
                ),
            ],
            id="group-six_compound-unzip+convert",
        ),
        pytest.param(
            "field_strength",
            True,
            False,
            marks=pytest.mark.slow,
            id="group-field_strength-unzip_only",
        ),
        pytest.param(
            "chronic",
            True,
            True,
            marks=[
                pytest.mark.slow,
                pytest.mark.skipif(
                    not _have_dicom2nifti,
                    reason="dicom2nifti not installed – skipping conversion test",
                ),
            ],
            id="group-chronic-unzip+convert",
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
    Exercise :pyfunc:`miblab.rat_fetch` across single-study and group cases.
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


if __name__ == "__main__":
    out_dir = Path.cwd() / "rat_data"   # persists on disk (manual run)
    out_dir.mkdir(exist_ok=True)

    # Single study, full pipeline:
    # rat_fetch("S01", folder=out_dir, unzip=True, convert=True)

    # Or groups (set convert=False for a quicker smoke run):
    rat_fetch("rifampicin_effect_size", folder=out_dir, unzip=True, convert=True)

    print(f"[OK] Data saved in: {out_dir}")