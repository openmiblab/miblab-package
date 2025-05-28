import os
from tqdm import tqdm

try:
    from totalsegmentator import map_to_binary
    from totalsegmentator.python_api import totalsegmentator
    totalsegmentor_installed = True
except ImportError:
    totalsegmentor_installed = False

try:
    import vreg
    vreg_installed = True
except ImportError:
    vreg_installed = False


TMPPATH = os.getcwd()


def _totseg(vol, cutoff=None, task='total', roi_subset=None, **kwargs):

    print('Saving source as nifti..')
    nifti_file = os.path.join(TMPPATH, 'source.nii.gz')
    vreg.write_nifti(vol, nifti_file)

    print('Segmenting organs..')
    totalsegmentator(nifti_file, TMPPATH, task=task, roi_subset=roi_subset, **kwargs)
    os.remove(nifti_file)

    if roi_subset is None:
        roi_set = list(map_to_binary.class_map[task].values())
    else:
        roi_set = roi_subset

    mask = {}
    for roi in tqdm(roi_set, desc='Reading results..'):
        roifile = os.path.join(TMPPATH, roi + '.nii.gz')
        v = vreg.read_nifti(roifile)
        if cutoff is not None:
            values = v.values
            values[values > cutoff] = 1
            values[values <= cutoff] = 0
            v.set_values(values)
        mask[roi] = v
        os.remove(roifile) 

    return mask


def totseg(vol, cutoff=None, **kwargs): 
    """Run totalsegmentator on one or more volumes.

    Source: `totalsegmentator <https://github.com/wasserth/TotalSegmentator>`_.

    Args:
        vol (vreg.Volume3D or list): Either a single volume, or a list 
            of volumes to be segmented.
        cutoff (float, optional): Pixels with a probability higher 
            than cutoff will be included in the mask. If cutoff is 
            not provided, probabilities will be returned directly. 
            Defaults to None.
        kwargs: Any keyword arguments accepted by the `totalsegmentor 
            python API <https://github.com/wasserth/TotalSegmentator/tree/master?tab=readme-ov-file#totalsegmentator>`_.

    Returns:
        dict: 
            dictionary with keys the mask names (ROI labels) 
            and values the corresponding vreg.Volume3D objects.

    Example:
        Use a machine with a cpu to run the task 'total_mr' on a 
        single volume saved as a nifti file:

        >>> import miblab
        >>> import vreg
        >>> vol = vreg.read_nifti('path/to/volume.nii.gz')
        >>> mask = miblab.totseg(vol, cutoff=0.01, task='total_mr', device='cpu')
        >>> print(mask['liver'].values)
        [0 1 1 ... 0 0 0]
    """
    if not vreg_installed:
        raise ImportError(
            'vreg is not installed. Please install it with "pip install vreg".'
            'To install all dlseg options at once, install miblab as pip install miblab[dlseg].'
        )
    if not totalsegmentor_installed:
        raise ImportError(
            'totalsegmentator is not installed. Please install it with "pip install totalsegmentator".'
            'To install all dlseg options at once, install miblab as pip install miblab[dlseg].'
        )

    if not isinstance(vol, list):
        return _totseg(vol, cutoff, **kwargs)

    total = {}
    for v in tqdm(vol, desc='Segmenting volumes..'):
        mask = _totseg(v, **kwargs)
        for roi in mask:
            if roi in total:
                values = total[roi].values + mask[roi].values
                total[roi].set_values(values)
            else:
                total[roi] = mask[roi]

    for roi, v in tqdm(total.items(), desc='Combining results..'):
        values = v.values/len(vol)
        if cutoff is not None:
            values[values > cutoff] = 1
            values[values <= cutoff] = 0
        v.set_values(values)
        total[roi] = v

    return total