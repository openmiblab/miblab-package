import os
import sys
import numpy as np
from miblab.data import zenodo_fetch
from miblab.data import clear_cache_datafiles
import scipy.ndimage as ndi

if sys.version_info < (3, 9):
    # importlib.resources either doesn't exist or lacks the files()
    # function, so use the PyPI version:
    import importlib_resources
else:
    # importlib.resources has files(), so use that:
    import importlib.resources as importlib_resources

try:
    from monai.networks.nets.unetr import UNETR
    from monai.inferers import sliding_window_inference
    monai_installed = True

except ImportError:
    monai_installed = False

try:
    import torch
    torch_installed = True
except ImportError:
    torch_installed = False

MODEL = 'UNETR_kidneys_v2.pth'
MODEL_DOI = "15521814"


def kidney_pc_dixon(input_array, overlap=0.3, postproc=True, clear_cache = False, verbose=False):
    
    """
    Segment individual kidneys on post-contrast Dixon images.
     
    This requires 4-channel input data with out-phase images, 
    in-phase images, water maps, and fat maps.

    This uses a pretrained UNETR-based model in MONAI, hosted on 
    `Zenodo <https://zenodo.org/records/15521814>`_

    Args:
        input_array (numpy.ndarray): A 4D numpy array of shape 
            [x, y, z, contrast] representing the input medical image 
            volume. The last index must contain out-phase, in-phase, 
            water and fat images, in that order.
        overlap (float): defines the amount of overlap between 
            adjacent sliding window patches during inference. A 
            higher value (e.g., 0.5) improves prediction smoothness 
            at patch borders but increases computation time.
        postproc (bool): If True, applies post-processing to select 
            the largest connected component from the UNETR output 
            for each kidney mask
        clear_cache: If True, the downloaded pth file is removed 
            again after running the inference.
        verbose (bool): If True, prints logging messages.

    Returns:
        dict: 
            A dictionary with the keys 'leftkidney' and 
            'rightkidney', each containing a binary NumPy array 
            representing the respective kidney mask.
    Example:

        >>> import numpy as np
        >>> import miblab
        >>> data = np.random.rand((128, 128, 30, 4))
        >>> mask = miblab.kidney_pc_dixon(data)
        >>> print(mask['leftkidney'])
        [0 1 1 ... 0 0 0]
    """
    if not torch_installed:
        raise ImportError(
            'torch is not installed. Please install it with "pip install torch".'
            'To install all dlseg options at once, install miblab as pip install miblab[dlseg].'
        )
    if not monai_installed:
        raise ImportError(
            'totalsegmentator is not installed. Please install it with "pip install totalsegmentator".'
            'To install all dlseg options at once, install miblab as pip install miblab[dlseg].'
        )

    if verbose:
        print('Downloading model..')

    temp_dir = importlib_resources.files('miblab.datafiles')
    weights_path = zenodo_fetch(MODEL, temp_dir, MODEL_DOI)

    if verbose:
        print('Applying model to data..')

    # Setup device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # Define model architecture
    model = UNETR(
        in_channels=4,
        out_channels=3, # BACKGROUND, RIGHT KIDNEY (left on image), LEFT KIDNEY (right on image)
        img_size=(80, 80, 80),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        proj_type="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    ).to(device)

    input_array = np.transpose(input_array, (3, 0, 1, 2)) # from (x,y,z,c) to (c,y,x,z)

    # Normalize data
    input_array_out   = (input_array[0,...]-np.average(input_array[0,...]))/np.std(input_array[0,...])
    input_array_in    = (input_array[1,...]-np.average(input_array[1,...]))/np.std(input_array[1,...])
    input_array_water = (input_array[2,...]-np.average(input_array[2,...]))/np.std(input_array[2,...])
    input_array_fat   = (input_array[3,...]-np.average(input_array[3,...]))/np.std(input_array[3,...])

    input_array = np.stack((input_array_out, input_array_in, input_array_water, input_array_fat), axis=0)
    # Convert to NCHW[D] format: (1,c,y,x,z)
    # NCHW[D] stands for: batch N, channels C, height H, width W, depth D
    input_array = input_array.transpose(0,2,1,3) # from (x,y,z) to (y,x,z)
    input_array = np.expand_dims(input_array, axis=(0))

    # Convert to tensor
    input_tensor = torch.tensor(input_array)

    # Load model weights
    weights = torch.load(weights_path, map_location=device)
    model.load_state_dict(weights)
    model.eval() 

    with torch.no_grad():
        output_tensor = sliding_window_inference(input_tensor, (80,80,80), 4, model, overlap=overlap, device=device_str, progress=True) 

    if verbose:
        print('Post-processing results...')

    # From probabilities for each channel to label image
    output_tensor = torch.argmax(output_tensor, dim=1)

    # Convert to numpy
    output_array = output_tensor.numpy(force=True)[0,:,:,:]
        
    # Transpose to original shape
    output_array = output_array.transpose(1,0,2) #from (y,x,z) to (x,y,z)

    if postproc == True:
        left_kidney, right_kidney = _kidney_masks(output_array)

    else:
        left_kidney=output_array[output_array == 2]
        left_kidney[left_kidney==2]=1
        
        right_kidney=output_array[output_array == 1]

    kidneys = {
        "leftkidney": left_kidney,
        "rightkidney": right_kidney
    }

    if clear_cache:
        if verbose:
            print('Deleting downloaded files...')
        clear_cache_datafiles(temp_dir)

    return kidneys  


def _largest_cluster(array:np.ndarray)->np.ndarray:
    """Given a mask array, return a new mask array containing only the largesr cluster.

    Args:
        array (np.ndarray): mask array with values 1 (inside) or 0 (outside)

    Returns:
        np.ndarray: mask array with only a single connect cluster of pixels.
    """
    # Label all features in the array
    label_img, cnt = ndi.label(array)
    # Find the label of the largest feature
    labels = range(1,cnt+1)
    size = [np.count_nonzero(label_img==l) for l in labels]
    max_label = labels[size.index(np.amax(size))]
    # Return a mask corresponding to the largest feature
    return label_img==max_label

def _kidney_masks(output_array:np.ndarray)->tuple:
    """Extract kidney masks from the output array of the UNETR

    Args:
        output_array (np.ndarray): 3D numpy array (x,y,z) with integer labels (0=background, 1=right kidney, 2=left kidney)

    Returns:
        tuple: A tuple of 3D numpy arrays (left_kidney, right_kidney) with masks for the kidneys.
    """
    left_kidney = _largest_cluster(output_array == 2)
    right_kidney = _largest_cluster(output_array == 1)

    return left_kidney, right_kidney