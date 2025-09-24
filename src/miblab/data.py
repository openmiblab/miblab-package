from __future__ import annotations
import os
import zipfile
import subprocess
import shutil
import importlib.util
from pathlib import Path
from typing import List

# Optional-dependency detection
_have_requests      = importlib.util.find_spec("requests")      is not None
_have_tqdm          = importlib.util.find_spec("tqdm")          is not None
_have_dicom2nifti   = importlib.util.find_spec("dicom2nifti")   is not None
_have_osfclient     = importlib.util.find_spec("osfclient")   is not None 

if _have_requests:
    import requests                           # type: ignore
    from requests.adapters import HTTPAdapter, Retry
    _rat_session = requests.Session()
    _rat_session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=3,
                backoff_factor=1,             # 1 s → 2 s → 4 s
                status_forcelist=(502, 503, 504),
            )
        ),
    )
else:      # pragma: no cover – network code unusable without requests anyway
    _rat_session = None                       # type: ignore

if _have_tqdm:
    from tqdm import tqdm                     # type: ignore
if _have_dicom2nifti:
    try:
        import dicom2nifti                   # type: ignore
    except Exception:                         # wheel import error, etc.
        _have_dicom2nifti = False
if _have_osfclient:
    from osfclient.api import OSF  # type: ignore[import-not-found]
else:
    from typing import Any
    OSF = Any  # type: ignore[assignment]

#  Unified flag for “any required extra-dependency is missing”
import_error = not (_have_requests and _have_tqdm)

# Zenodo DOI of the repository
DOI = {
    'MRR': "15285017",    
    'TRISTAN': "15301607",
    'RAT': "17178063",
}

# miblab datasets
DATASETS = {
    'KRUK.dmr.zip': {'doi': DOI['MRR']},
    'tristan_humans_healthy_controls.dmr.zip': {'doi': DOI['TRISTAN']},
    'tristan_humans_healthy_ciclosporin.dmr.zip': {'doi': DOI['TRISTAN']},
    'tristan_humans_healthy_metformin.dmr.zip': {'doi': DOI['TRISTAN']},
    'tristan_humans_healthy_rifampicin.dmr.zip': {'doi': DOI['TRISTAN']},
    'tristan_humans_patients_rifampicin.dmr.zip': {'doi': DOI['TRISTAN']},
    'tristan_rats_healthy_multiple_dosing.dmr.zip': {'doi': DOI['TRISTAN']},
    'tristan_rats_healthy_reproducibility.dmr.zip': {'doi': DOI['TRISTAN']},
    'tristan_rats_healthy_six_drugs.dmr.zip': {'doi': DOI['TRISTAN']},
}

# ─────────────────────────────────────────────────────────────────────────────
# ADDED: Study groups + helpers 
# ─────────────────────────────────────────────────────────────────────────────
#: Mapping of friendly **group names** to lists of study IDs.
#: 
#: Groups:
#: - ``rifampicin_effect_size`` → S01, S02, S03, S04
#: - ``six_compound``           → S05, S06, S07, S08, S09, S10, S12
#: - ``field_strength``         → S13
#: - ``chronic``                → S11, S14, S15
#:
#: These names are accepted by :pyfunc:`rat_fetch` in addition to single
#: study IDs (e.g. ``"S03"``) and ``"all"``.
_GROUPS = {
    "rifampicin_effect_size": ["s01", "s02", "s03", "s04"],
    "six_compound":           ["s05", "s06", "s07", "s08", "s09","s10", "s12"],
    "field_strength":         ["s13"],
    "chronic":                ["s11", "s14", "s15"],
}

def _norm_key(name: str | None) -> str:
    """
    Normalize user input to a token suitable for matching.

    * Lowercase
    * Replace any non-alphanumeric char with underscore
    * Strip leading/trailing underscores

    Examples
    --------
    >>> _norm_key("Six Compound")
    'six_compound'
    >>> _norm_key("Rifampicin-Effect-Size")
    'rifampicin_effect_size'
    """
    if name is None:
        return ""
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")

# ─────────────────────────────────────────────────────────────────────────────

def zenodo_fetch(dataset: str, folder: str, doi: str = None, filename: str = None,
                 extract: bool = False, verbose: bool = False):
    """Download a dataset from Zenodo.

    Note if a dataset already exists locally it will not be downloaded 
    again and the existing file will be returned. 

    Args:
        dataset (str): Name of the dataset
        folder (str): Local folder where the result is to be saved
        doi (str, optional): Digital object identifier (DOI) of the 
          Zenodo repository where the dataset is uploaded. If this 
          is not provided, the function will look for the dataset in
          miblab's own Zenodo repositories.
        filename (str, optional): Filename of the downloaded dataset. 
          If this is not provided, then *dataset* is used as filename.
        extract (bool): Whether to automatically extract downloaded ZIP files. 
        verbose (bool): If True, prints logging messages.

    Raises:
        NotImplementedError: If miblab is not installed with the data
          option.
        requests.exceptions.ConnectionError: If the connection to 
          Zenodo cannot be made.

    Returns:
        str: Full path to the downloaded datafile.
    """
    if import_error:
        raise NotImplementedError(
            'Please install miblab as pip install miblab[data] '
            'to use this function.'
        )
        
    # Create filename 
    if filename is None:
        file = os.path.join(folder, dataset)
    else:
        file = os.path.join(folder, filename)

    # If it is not already downloaded, download it.
    if os.path.exists(file):
        if verbose:
            print(f"Skipping {dataset} download, file {file} already exists.")
    else:
        # Get DOI
        if doi is None:
            if dataset in DATASETS:
                doi = DATASETS[dataset]['doi']
            else:
                raise ValueError(
                    f"{dataset} does not exist in one of the miblab "
                    f"repositories on Zenodo. If you want to fetch " 
                    f"a dataset in an external Zenodo repository, please "
                    f"provide the doi of the repository."
                )
        
        # Dataset download link
        file_url = f"https://zenodo.org/records/{doi}/files/{filename or dataset}"

        # Make the request and check for connection error
        try:
            file_response = requests.get(file_url) 
        except requests.exceptions.ConnectionError as err:
            raise requests.exceptions.ConnectionError(
                f"\n\n"
                f"A connection error occurred trying to download {dataset} "
                f"from Zenodo. This usually happens if you are offline. "
                f"The detailed error message is here: {err}"
            ) 
        
        # Check for other errors
        file_response.raise_for_status()

        # Create the folder if needed
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Save the file
        with open(file, 'wb') as f:
            f.write(file_response.content)

    # If the zip file is requested we are done
    if not extract:
        return file
    
    # If extraction requested, returned extracted
    if file[-4:] == '.zip':
        extract_to = file[:-4]
    else:
        extract_to = file + '_unzip'

    # Skip extraction if the folder already exists
    if os.path.exists(extract_to):
        if verbose:
            print(f"Skipping {file} extraction, folder {extract_to} already exists.")
        return extract_to

    # Perform extraction
    os.makedirs(extract_to)
    with zipfile.ZipFile(file, 'r') as zip_ref:
        bad_file = zip_ref.testzip()
        if bad_file:
            raise zipfile.BadZipFile(
                f"Cannot extract: corrupt file {bad_file}."
            )
        zip_ref.extractall(extract_to)

    return extract_to

    
def clear_cache_datafiles(directory: str, verbose: bool = True):
    """
    Delete all files and subdirectories in the specified cache directory,
    except for '__init__' files.

    Args:
        directory (str): Path to the directory to clear.
        verbose (bool): If True, prints names of deleted items.

    Raises:
        FileNotFoundError: If the directory does not exist.
        OSError: If a file or folder cannot be deleted.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    deleted = []
    for item in os.listdir(directory):
        path = os.path.join(directory, item)

        # Skip __init__ files (e.g., __init__.py, __init__.pyc)
        if os.path.isfile(path) and os.path.splitext(item)[0] == '__init__':
            continue

        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
                deleted.append(path)
                if verbose:
                    print(f"Deleted file: {path}")
            elif os.path.isdir(path):
                shutil.rmtree(path)
                deleted.append(path)
                if verbose:
                    print(f"Deleted folder: {path}")
        except Exception as e:
            print(f"Error deleting {path}: {e}")

    if verbose and not deleted:
        print("Directory is already clean.")

def osf_fetch(dataset: str, folder: str, project: str = "un5ct", token: str = None, extract: bool = True, verbose: bool = True):
    """
    Download a dataset from OSF (Open Science Framework).

    This function downloads a specific dataset (folder or subfolder) from a public or private OSF project.
    Files are saved into the specified local directory. If a zip file is found, it will be extracted by default.

    Args:
        dataset (str): Subfolder path inside the OSF project. If an empty string, all files in the root will be downloaded (use with caution).
        folder (str): Local folder where the dataset will be saved.
        project (str, optional): OSF project ID (default is "un5ct").
        token (str, optional): Personal OSF token for accessing private projects. Read from OSF_TOKEN environment variable if needed.
        extract (bool, optional): Whether to automatically unzip downloaded .zip files (default is True).
        verbose (bool, optional): Whether to print progress messages (default is True).

    Raises:
        FileNotFoundError: If the specified dataset path does not exist in the OSF project.
        NotImplementedError: If required packages are not installed.

    Returns:
        str: Path to the local folder containing the downloaded data.

    Example:
        >>> from miblab.data import osf_fetch
        >>> osf_fetch('TRISTAN/RAT/bosentan_highdose/Sanofi', 'test_download')
    """
    if import_error or not _have_osfclient:
        raise NotImplementedError(
            "Please install miblab[data] (includes osfclient, requests, tqdm) to use this function."
        )

    # Prepare local folder
    os.makedirs(folder, exist_ok=True)

    # Connect to OSF and locate project storage
    osf = OSF(token=token)  #osf = OSF()  for public projects
    project = osf.project(project)
    storage = project.storage('osfstorage')

    # Navigate the dataset folder if provided
    current = storage
    if dataset:
        parts = dataset.strip('/').split('/')
        for part in parts:
            for f in current.folders:
                if f.name == part:
                    current = f
                    break
            else:
                raise FileNotFoundError(f"Folder '{part}' not found when navigating path '{dataset}'.")

    # Recursive download of all files and folders
    def download(current_folder, local_folder):
        os.makedirs(local_folder, exist_ok=True)
        files = list(current_folder.files)
        iterator = tqdm(files, desc=f"Downloading to {local_folder}") if verbose and files else files
        for file in iterator:
            local_file = os.path.join(local_folder, file.name)
            try:
                with open(local_file, 'wb') as f:
                    file.write_to(f)
            except Exception as e:
                if verbose:
                    print(f"Warning downloading {file.name}: {e}")

        for subfolder in current_folder.folders:
            download(subfolder, os.path.join(local_folder, subfolder.name))

    download(current, folder)

    # Extract all downloaded zip files if needed
    if extract:
        for dirpath, _, filenames in os.walk(folder):
            for filename in filenames:
                if filename.lower().endswith('.zip'):
                    zip_path = os.path.join(dirpath, filename)
                    extract_to = os.path.join(dirpath, filename[:-4])
                    os.makedirs(extract_to, exist_ok=True)
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            bad_file = zip_ref.testzip()
                            if bad_file:
                                raise zipfile.BadZipFile(f"Corrupt file {bad_file} inside {zip_path}")
                            zip_ref.extractall(extract_to)
                        os.remove(zip_path)
                        if verbose:
                            print(f"Unzipped and deleted {zip_path}")
                    except Exception as e:
                        if verbose:
                            print(f"Warning unzipping {zip_path}: {e}")
    return folder


def osf_upload(folder: str, dataset: str, project: str = "un5ct", token: str = None, verbose: bool = True, overwrite: bool = True):
    """
    Upload a file to OSF (Open Science Framework) using osfclient.

    This function uploads a single local file to a specified path inside an OSF project.
    Intermediate folders must already exist in the OSF project; osfclient does not create them.
    If the file already exists, it can be overwritten or skipped.

    Args:
        folder (str): Path to the local file to upload.
        dataset (str): OSF path where the file should be placed (e.g., "Testing/filename.txt").
        project (str): OSF project ID (default: "un5ct").
        token (str): OSF personal token for private/write access.
        verbose (bool): Whether to print progress messages (default True).
        overwrite (bool): Whether to replace an existing file if it already exists (default True).

    Raises:
        FileNotFoundError: If the file does not exist.
        NotImplementedError: If osfclient is not installed.
        RuntimeError: If upload fails for any reason.

    Example:
        >>> from miblab import osf_upload
        >>> osf_upload(
        ...     folder='data/results.csv',
        ...     dataset='Testing/results.csv',
        ...     project='un5ct',
        ...     token='your-osf-token',
        ...     verbose=True,
        ...     overwrite=True
        ... )
    """
    import os

    # Check that optional dependencies are installed
    if import_error:
        raise NotImplementedError("Please install miblab[data] to use this function.")

    # Check that the specified local file exists
    if not os.path.isfile(folder):
        raise FileNotFoundError(f"Local file not found: {folder}")

    # Authenticate and connect to the OSF project
    osf = OSF(token=token)
    project = osf.project(project)
    storage = project.storage("osfstorage")

    # Clean and prepare the remote dataset path
    full_path = dataset.strip("/")

    # Check if the file already exists on OSF
    existing = next((f for f in storage.files if f.path == "/" + full_path), None)
    if existing:
        if overwrite:
            if verbose:
                print(f"File '{full_path}' already exists. Deleting before re-upload...")
            try:
                existing.remove()
            except Exception as e:
                raise RuntimeError(f"Failed to delete existing file before overwrite: {e}")
        else:
            if verbose:
                print(f"File '{full_path}' already exists. Skipping (overwrite=False).")
            return

    # Upload the file
    size_mb = os.path.getsize(folder) / 1e6
    with open(folder, "rb") as f:
        if verbose:
            print(f"Uploading '{os.path.basename(folder)}' ({size_mb:.2f} MB) to '{full_path}'...")
        try:
            storage.create_file(full_path, f)
            if verbose:
                print("Upload complete.")
        except Exception as e:
            raise RuntimeError(f"Failed to upload file: {e}")

#  Utilities
#  Utilities
def _unzip_nested(zip_path: str | Path, extract_to: str | Path,
                  *, keep_archives: bool = False) -> None:
    """
    Recursively extract *every* ZIP found inside *zip_path*.

    Parameters
    ----------
    zip_path
        Path to the outer **.zip** file downloaded from Zenodo.
    extract_to
        Target directory.  It is created if it does not exist.
    keep_archives
        • *False* (default) → **delete** each inner archive after it has
          been unpacked, leaving only the extracted folders/files.  
        • *True*  → preserve the nested ``.zip`` files for checksum /
          forensic work.

    Notes
    -----
    * The routine is **pure-Python** (built-in ``zipfile``); no external
      7-Zip dependency.  
    * Extraction is breadth-first: after the outer ZIP is unpacked, the
      function scans the new tree for ``*.zip`` and repeats until none
      remain.  
    * Corrupt inner archives are caught and logged to *stdout* but do
      **not** abort the entire operation.

    Additional handling (added)
    ---------------------------
    * Skips macOS Finder metadata that may masquerade as ZIPs:
      ``__MACOSX/`` folders and ``._<name>.zip`` files (resource forks).
      These are not real archives and would otherwise trigger repeated
      “File is not a zip file” warnings.
    * Tracks already-processed inner ZIPs so a failing file is not retried
      on subsequent passes.

    Examples
    --------
    >>> _unzip_nested("S03.zip", "S03_unzipped", keep_archives=True)
    """

    zip_path, extract_to = Path(zip_path), Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)

    # NEW: remember inner ZIPs we've already attempted
    processed: set[Path] = set()

    while True:
        # NEW: collect candidate ZIPs, skipping macOS metadata & already processed ones
        inners: list[Path] = []
        for p in extract_to.rglob("*.zip"):
            if p in processed:
                continue
            # Skip macOS Finder cruft: __MACOSX folders and resource-fork files "._*"
            if "__MACOSX" in p.parts or p.name.startswith("._"):
                processed.add(p)
                # Optionally remove tiny resource-fork "._*.zip" files to declutter
                try:
                    if p.name.startswith("._") and p.is_file():
                        p.unlink(missing_ok=True)
                except OSError:
                    pass
                continue
            inners.append(p)

        if not inners:
            break

        for inner in inners:
            processed.add(inner)  # mark first to avoid repeated attempts
            dest = inner.with_suffix("")      # “…/file.zip” → “…/file/”
            dest.mkdir(exist_ok=True)
            try:
                with zipfile.ZipFile(inner) as izf:
                    izf.extractall(dest)
                if not keep_archives:
                    inner.unlink()
            except zipfile.BadZipFile as exc:         # noqa: BLE001
                print(f"[rat_fetch] WARNING – cannot unzip {inner}: {exc}")

def _convert_dicom_to_nifti(source_dir: Path, output_dir: Path) -> None:
    """
    Convert *all* DICOM series found in *source_dir* to compressed NIfTI.

    A thin, tolerant wrapper around
    :pyfunc:`dicom2nifti.convert_directory`.  Any conversion error
    (corrupt slice, unsupported orientation, etc.) is printed and the
    function returns so the calling loop can continue with the next
    subject / day.

    Parameters
    ----------
    source_dir
        Directory that contains one or more DICOM series.
    output_dir
        Destination directory.  Created if missing.
        Each converted series is written as ``series_<UID>.nii.gz``.
    
    Examples
    --------
    >>> from pathlib import Path
    >>> _convert_dicom_to_nifti(Path("S01/Rat03/Day1/dicom"), Path("S01_nifti/Rat03/Day1"))
    """

    if not _have_dicom2nifti:
        raise NotImplementedError(
            "dicom2nifti is required for DICOM → NIfTI conversion."
        )
    
    # Only consider plain files directly under the series folder
    dcm_files = [p for p in source_dir.glob("*.dcm") if p.is_file()]
    if len(dcm_files) < 3:
        # keep it quiet but informative
        print(f"[rat_fetch] SKIP – {source_dir} has only {len(dcm_files)} DICOM slice(s)")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        dicom2nifti.convert_directory(
            str(source_dir), str(output_dir), reorient=True
        )
    except Exception as exc:                          # noqa: BLE001
        print(f"[rat_fetch] ERROR – conversion failed for {source_dir}: {exc}")

def _relax_dicom2nifti_validators() -> None:
    """
    Disable dicom2nifti's strict slice-geometry validators.

    Pre-clinical (small-animal) scanners often produce DICOMs that fail
    dicom2nifti’s default **orthogonality** / **slice-increment** checks
    even though the data reconstructs fine.  This helper tries to import
    ``dicom2nifti.settings`` and, if present, toggles every
    *disable_validate_* flag known across versions 2 → 3.

    The call is **idempotent** – safe to invoke multiple times.

    No error is raised when *dicom2nifti* is not installed; the caller
    should already have checked the `_have_dicom2nifti` feature-flag.
    """

    try:
        import dicom2nifti.settings as _dset          # type: ignore
    except ModuleNotFoundError:
        return
    for fn in ("disable_validate_orthogonal",
               "disable_validate_sliceincrement",
               "disable_validate_slice_increment",
               "disable_validate_dimensions",
               "disable_validate_dimension",
               "disable_validate_slicecount",
               "disable_validate_slice_count",):
        if hasattr(_dset, fn):
            getattr(_dset, fn)()

#  Public TRISTAN RAT Download Zenodo API
def rat_fetch(
    dataset: str | None = None,
    *,
    folder: str | Path = "./tristanrat",
    unzip: bool  = True,
    convert: bool = False,
    keep_archives: bool = False,
) -> List[str]:
    """
    Download, recursively extract, and (optionally) convert TRISTAN rat
    MRI studies from Zenodo (record **17178063**).

    The helper understands the 15 published studies **S01 … S15**.  
    Pass ``dataset="all"`` (or leave *dataset* empty) to fetch them all.

    Parameters
    ----------
    dataset
        ``"S01" … "S15"`` to grab a single study  
        ``"all"`` or *None* to fetch them all  
        **or one of the group names below**.
    folder
        Root directory that will hold the ``SXX.zip`` files and the
        extracted DICOM tree.  A sibling directory
        ``<folder>_nifti/`` is used for conversion output.
    unzip
        If *True*, each ZIP is unpacked **recursively** (handles inner
        ZIP-in-ZIP structures).
    convert
        If *True*, every DICOM folder is converted to compressed NIfTI
        (_requires the **dicom2nifti** wheel and ``unzip=True``_).
    keep_archives
        Forwarded to :func:`_unzip_nested`; set *True* to retain each
        inner ZIP after extraction (useful for auditing).

    Study groups
    ------------
    In addition to single IDs, you can pass one of these **group names**:

    - ``rifampicin_effect_size`` → S01, S02, S03, S04
    - ``six_compound``           → S05, S06, S07, S08, S09, S10, S12
    - ``field_strength``         → S13
    - ``chronic``                → S11, S14, S15

    Examples
    --------
    **Download a single study and leave it zipped**

    >>> from miblab import rat_fetch
    >>> rat_fetch("S01", folder="~/tristanrat", unzip=False)
    ['/home/you/tristanrat/S01.zip']

    **Fetch the entire collection, unzip, but skip conversion**

    >>> rat_fetch(dataset="all",
    ...           folder="./rat_data",
    ...           unzip=True,
    ...           convert=False)

    **Full end-to-end pipeline (requires dicom2nifti)**

    >>> rat_fetch("S03",
    ...           folder="./rat_data",
    ...           unzip=True,
    ...           convert=True)

    **Use a study group (e.g. six compounds)**

    >>> rat_fetch("six_compound", folder="./rat_data", unzip=True, convert=True)

    **Other groups**

    >>> rat_fetch("rifampicin_effect_size", folder="./rat_data", unzip=True)
    >>> rat_fetch("field_strength", folder="./rat_data", unzip=True, convert=False)
    >>> rat_fetch("chronic", folder="./rat_data", unzip=True, convert=True)

    The call returns the list of ZIP paths; side-effects are files
    extracted (and optionally NIfTI volumes) under *folder*.
    """
    # ── dependency guards ───────────────────────────────────────────────────
    if not _have_requests:
        raise NotImplementedError(
            "rat_fetch needs the optional 'requests' extra "
            "(pip install miblab[data])."
        )
    if convert and not unzip:
        raise ValueError("convert=True requires unzip=True.")

    # ── ADDED: pre-handle group names (without removing original logic) ─────
    _group_token = None
    _token = _norm_key(dataset or "all")
    if _token in _GROUPS:
        # remember which group was requested; we'll override `studies` later
        _group_token = _token
        # set dataset to a valid single ID to avoid the original resolver raising
        dataset = "S01"

    # ── resolve study IDs (original block kept intact) ──────────────────────
    dataset = (dataset or "all").lower()
    valid_ids = [f"s{i:02d}" for i in range(1, 16)]   # S01 … S15 only
    if dataset == "all":
        studies = valid_ids
    elif dataset in valid_ids:
        studies = [dataset]
    else:
        raise ValueError(
            f"Unknown study '{dataset}'. Choose one of "
            f"{', '.join(valid_ids)} or 'all'."
        )

    # ── ADDED: if a group was requested, override `studies` here ────────────
    if _group_token is not None:
        studies = _GROUPS[_group_token]

    # ── local paths & URL template ──────────────────────────────────────────
    folder     = Path(folder).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    nifti_root = folder.parent / f"{folder.name}_nifti"
    base_url   = f"https://zenodo.org/api/records/{DOI['RAT']}/files"

    downloaded: List[str] = []

    # ── download loop ───────────────────────────────────────────────────────
    desc = "Downloading TRISTAN rat studies" if _have_tqdm else None
    it   = tqdm(studies, desc=desc, leave=False) if _have_tqdm else studies

    for sid in it:
        zip_name = f"{sid.upper()}.zip"
        zip_path = folder / zip_name
        url      = f"{base_url}/{zip_name}/content"

        # skip if already present
        if not zip_path.exists():
            try:
                with _rat_session.get(url, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with open(zip_path, "wb") as fh:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            fh.write(chunk)
            except Exception as exc:                   # noqa: BLE001
                print(f"[rat_fetch] WARNING – could not download {zip_name}: {exc}")
                continue
        downloaded.append(str(zip_path))

        # ── extraction ───────────────────────────────────────
        if unzip:
            study_dir = folder / sid.upper()
            _unzip_nested(zip_path, study_dir, keep_archives=keep_archives)

            # ── optional DICOM ➜ NIfTI ──────────────────────
            if convert:
                _relax_dicom2nifti_validators()
                for dcm_dir in study_dir.rglob("*"):
                    if not dcm_dir.is_dir():
                        continue
                    # ── change this condition ──
                    dcms = [p for p in dcm_dir.glob("*.dcm") if p.is_file()]
                    if len(dcms) >= 3:   # <- was >= 1
                        rel_out = dcm_dir.relative_to(folder)
                        _convert_dicom_to_nifti(dcm_dir, nifti_root / rel_out)

    return downloaded