import os
import zipfile
import subprocess
import shutil

# Try importing optional dependencies
try:
    import requests
    from osfclient.api import OSF
    from tqdm import tqdm
    import_error = False
except ImportError:
    import_error = True

# Zenodo DOI of the repository
DOI = {
    'MRR': "15285017",    
    'TRISTAN': "15301607", 
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
        >>> from miblab import osf_fetch
        >>> osf_fetch('TRISTAN/RAT/bosentan_highdose/Sanofi', 'test_download')
    """
    if import_error:
        raise NotImplementedError(
            "Please install miblab as pip install miblab[data] to use this function."
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
    from osfclient.api import OSF
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