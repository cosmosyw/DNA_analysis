### adapted from ImageAnalysis3 - for alignment of images based on fiducial beads
import os
import numpy as np
from skimage.registration import phase_cross_correlation
from scipy.spatial.distance import pdist, squareform, euclidean
import math
import cv2
import time
from scipy.ndimage.interpolation import map_coordinates
import pickle


def _find_boundary(_ct, _radius, _im_size):
    _bds = []
    for _c, _sz in zip(_ct, _im_size):
        _bds.append([max(_c-_radius, 0), min(_c+_radius, _sz)])
    
    return np.array(_bds, dtype=int)


def generate_drift_crops(single_im_size, coord_sel=None, drift_size=None):
    """Function to generate drift crop from a selected center and given drift size
    keywards:
        single_im_size: single image size to generate crops, np.ndarray like;
        coord_sel: selected center coordinate to split image into 4 rectangles, np.ndarray like;
        drift_size: size of drift crop, int or np.int;
    returns:
        crops: 4x3x2 np.ndarray. 
    """
    # check inputs
    _single_im_size = np.array(single_im_size)
    if coord_sel is None:
        coord_sel = np.array(_single_im_size/2, dtype=int)
    if coord_sel[-2] >= _single_im_size[-2] or coord_sel[-1] >= _single_im_size[-1]:
        raise ValueError(f"wrong input coord_sel:{coord_sel}, should be smaller than single_im_size:{single_im_size}")
    if drift_size is None:
        drift_size = int(np.max(_single_im_size)/4)
        
    # generate crop centers
    crop_cts = [
        np.array([coord_sel[-3]/2, 
                  coord_sel[-2]/2, 
                  coord_sel[-1]/2,]),
        np.array([coord_sel[-3]/2, 
                  (coord_sel[-2]+_single_im_size[-2])/2, 
                  (coord_sel[-1]+_single_im_size[-1])/2,]),
        np.array([coord_sel[-3]/2, 
                  (coord_sel[-2]+_single_im_size[-2])/2, 
                  coord_sel[-1]/2,]),
        np.array([coord_sel[-3]/2, 
                  coord_sel[-2]/2, 
                  (coord_sel[-1]+_single_im_size[-1])/2,]),
        np.array([coord_sel[-3]/2, 
                  coord_sel[-2], 
                  coord_sel[-1]/2,]),
        np.array([coord_sel[-3]/2, 
                  coord_sel[-2], 
                  (coord_sel[-1]+_single_im_size[-1])/2,]),
        np.array([coord_sel[-3]/2, 
                  coord_sel[-2]/2, 
                  coord_sel[-1],]),
        np.array([coord_sel[-3]/2, 
                  (coord_sel[-2]+_single_im_size[-2])/2, 
                  coord_sel[-1],]),                               
    ]
    # generate boundaries
    crops = [_find_boundary(_ct, _radius=drift_size/2, _im_size=single_im_size) for _ct in crop_cts]
        
    return np.array(crops)

def shift_spots(spots, drift):
    """shift the spots based on drift"""
    if np.shape(spots)[1] == 11: # this means 3d fitting result
        coords = np.array(spots).copy()[:,1:4]
    else:
        raise ValueError(f"Wrong input coords")
    # apply drift
    corr_coords = coords + drift
    # update spots
    output_coords = np.array(spots).copy()
    output_coords[:, 1:4] = corr_coords
    return output_coords

def microscope_translation_spot(spots, microscope_params, image_size = (50,2048,2048)):
    """Translate spots given microscope"""
    _fov_spots = spots.copy()
    # load microscope.json
    if _fov_spots.shape[1]==11:
        _coords = _fov_spots[:, 1:4]
    else:
        _coords = _fov_spots
    
    # transpose
    if 'transpose' in microscope_params and microscope_params['transpose']:
        _coords = _coords[:, np.array([0,2,1])]
    if 'flip_horizontal' in microscope_params and microscope_params['flip_horizontal']:
        _coords[:,2] = -1 * (_coords[:,2] - image_size[2]/2) + image_size[2]/2
    if  'flip_vertical' in microscope_params and microscope_params['flip_vertical']:
        _coords[:,1] = -1 * (_coords[:,1] - image_size[1]/2) + image_size[1]/2
    
    if _fov_spots.shape[1]==11:
        _fov_spots[:, 1:4] = _coords
    else:
        _fov_spots = _coords
    
    return _fov_spots

def reverse_microscope_translation_spot(spots, microscope_params, image_size = (50,2048,2048)):
    """Translate spots given microscope"""
    _fov_spots = spots.copy()
    # load microscope.json
    if _fov_spots.shape[1]==11:
        _coords = _fov_spots[:, 1:4]
    else:
        _coords = _fov_spots

    # flip
    if  'flip_vertical' in microscope_params and microscope_params['flip_vertical']:
        _coords[:,1] = -1 * (_coords[:,1] - image_size[1]/2) + image_size[1]/2
    if 'flip_horizontal' in microscope_params and microscope_params['flip_horizontal']:
        _coords[:,2] = -1 * (_coords[:,2] - image_size[2]/2) + image_size[2]/2
    # transpose
    if 'transpose' in microscope_params and microscope_params['transpose']:
        _coords = _coords[:, np.array([0,2,1])]
    
    if _fov_spots.shape[1]==11:
        _fov_spots[:, 1:4] = _coords
    else:
        _fov_spots = _coords
    
    return _fov_spots

def generate_polynomial_data(coords, max_order):
    """function to generate polynomial data
    Args:
        coords: coordinates, np.ndarray, n_points by n_dimensions
        max_order: maximum order of polynomial, int
    Return:
        _X: data for polynomial, n_points by n_columns
    """
    import itertools
    _X = []
    for _order in range(int(max_order)+1):
        for _lst in itertools.combinations_with_replacement(
                coords.transpose(), _order):
            # initialize one column
            _xi = np.ones(np.shape(coords)[0])
            # calculate product
            for _v in _lst:
                _xi *= _v
            # append
            _X.append(_xi)
    # transpose to n_points by n_columns
    _X = np.array(_X).transpose()
    
    return _X

def generate_chromatic_function(chromatic_const_file, drift=None):
    """Function to generate a chromatic abbrevation translation function from
    _const.pkl file"""

    if isinstance(chromatic_const_file, dict):
        _info_dict = {_k:_v for _k,_v in chromatic_const_file.items()}
    elif isinstance(chromatic_const_file, str):
        _info_dict = pickle.load(open(chromatic_const_file, 'rb'))
    elif chromatic_const_file is None:
        if drift is None:
            #print('empty_function')
            def _shift_function(_coords, _drift=drift): 
                return _coords
            return _shift_function
        else:
            _info_dict ={
                'constants': [np.array([0]) for _dft in drift],
                'fitting_orders': np.zeros(len(drift),dtype=np.int),
                'ref_center': np.zeros(len(drift)),
            }
    else:
        raise TypeError(f"Wrong input chromatic_const_file")

    # extract info
    _consts = _info_dict['constants']
    _fitting_orders = _info_dict['fitting_orders']
    _ref_center = _info_dict['ref_center']
    # drift
    if drift is None:
        _drift = np.zeros(len(_ref_center))
    else:
        _drift = drift[:len(_ref_center)]
    
    def _shift_function(_coords, _drift=_drift, 
                        _consts=_consts, 
                        _fitting_orders=_fitting_orders, 
                        _ref_center=_ref_center,
                        ):
        """generated translation function with constants and drift"""
        # return empty if thats the case
        if len(_coords) == 0:
            return _coords
        else:
            _coords = np.array(_coords)

        if np.shape(_coords)[1] == len(_ref_center):
            _new_coords = np.array(_coords).copy()
        elif np.shape(_coords)[1] == 11: # this means 3d fitting result
            _new_coords = np.array(_coords).copy()[:,1:1+len(_ref_center)]
        else:
            raise ValueError(f"Wrong input coords")

        _shifts = []
        for _i, (_const, _order) in enumerate(zip(_consts, _fitting_orders)):
            # calculate dX
            _X = generate_polynomial_data(_new_coords- _ref_center[np.newaxis,:], 
                                          _order)
            # calculate dY
            _dy = np.dot(_X, _const)
            _shifts.append(_dy)
        _shifts = np.array(_shifts).transpose()

        # generate corrected coordinates
        _corr_coords = _new_coords - _shifts + _drift

        # return as input
        if np.shape(_coords)[1] == len(_ref_center):
            _output_coords = _corr_coords
        elif np.shape(_coords)[1] == 11: # this means 3d fitting result
            _output_coords = np.array(_coords).copy()
            _output_coords[:,1:1+len(_ref_center)] = _corr_coords
        return _output_coords
    
    # return function
    return _shift_function


def align_image(
    src_im:np.ndarray, 
    ref_im:np.ndarray, 
    crop_list=None,
    precision_fold=100, 
    min_good_drifts=3, drift_diff_th=1., drift_pixel_threshold = 150, z_drift_threshold = 3):
    """Function to align one image by either FFT or spot_finding
        both source and reference images should be corrected
    """

    single_im_size = src_im.shape
    # check crop_list:
    if crop_list is None:
        crop_list = generate_drift_crops(single_im_size)
    for _crop in crop_list:
        if np.shape(np.array(_crop)) != (3,2):
            raise IndexError(f"crop should be 3x2 np.ndarray.")
    
    # define result flag
    _result_flag = 0
    
    if np.shape(src_im) != np.shape(ref_im):
        raise IndexError(f"shape of target image:{np.shape(src_im)} and reference image:{np.shape(ref_im)} doesnt match!")

    ## crop images
    _crop_src_ims, _crop_ref_ims = [], []
    for _crop in crop_list:
        _s = tuple([slice(*np.array(_c,dtype=int)) for _c in _crop])
        _crop_src_ims.append(src_im[_s])
        _crop_ref_ims.append(ref_im[_s])

    ## align two images
    _drifts = []
    for _i, (_sim, _rim) in enumerate(zip(_crop_src_ims, _crop_ref_ims)):
        # calculate drift with autocorr
        _dft, _error, _phasediff = phase_cross_correlation(_rim, _sim, 
                                                               upsample_factor=precision_fold)
        # append if the drift calculated pass certain criteria
        if (len(np.where(np.abs(_dft)>drift_pixel_threshold)[0])==0) & (len(np.where(_dft==0)[0])<=1) & (np.abs(_dft)[0]<z_drift_threshold):
            _drifts.append(_dft) 

        # detect variance within existing drifts
        _mean_dft = np.nanmean(_drifts, axis=0)
        if len(_drifts) >= min_good_drifts:
            _dists = np.linalg.norm(_drifts-_mean_dft, axis=1)
            _kept_drift_inds = np.where(_dists <= drift_diff_th)[0]
            if len(_kept_drift_inds) >= min_good_drifts:
                _updated_mean_dft = np.nanmean(np.array(_drifts)[_kept_drift_inds], axis=0)
                _result_flag = 'Optimal alignment'
                break
    
    # if no good drift and just one good drift is detected. Do it on the entire image
    if len(_drifts)==0:
        _dft, _error, _phasediff = phase_cross_correlation(ref_im, src_im, 
                                                               upsample_factor=precision_fold)
        return _dft, 'Failed alignment'
    elif len(_drifts)==1:
        _dft, _error, _phasediff = phase_cross_correlation(ref_im, src_im, 
                                                               upsample_factor=precision_fold)
        return _dft, 'Poor alignment'

    if '_updated_mean_dft' not in locals():
        _drifts = np.array(_drifts)
        # select top 3 drifts
        _dist_mat = squareform(pdist(_drifts))
        np.fill_diagonal(_dist_mat, np.inf)
        # select closest pair
        _sel_inds = np.array(np.unravel_index(np.argmin(_dist_mat), np.shape(_dist_mat)))
        _sel_drifts = list(_drifts[_sel_inds])
        # select closest 3rd drift
        third_drift = _drifts[np.argmin(_dist_mat[:, _sel_inds].sum(1))]
        diff_first_third = euclidean(third_drift, _sel_drifts[0])
        diff_second_third = euclidean(third_drift, _sel_drifts[1])
        diff_first_second = euclidean(_sel_drifts[0], _sel_drifts[1])
        _cv = np.std([diff_first_second, diff_first_third, diff_second_third])/np.mean([diff_first_second, diff_first_third, diff_second_third])
        if _cv<=0.5:
            _sel_drifts.append(_drifts[np.argmin(_dist_mat[:, _sel_inds].sum(1))])
        # return mean
        _updated_mean_dft = np.nanmean(_sel_drifts, axis=0)
        _result_flag = 'Suboptimal alignment'

    return  _updated_mean_dft, _result_flag

## alignment between re-labeled sample
def align_manual_points(pos_file_before, pos_file_after, verbose=True):
    """Function to align two manually picked position files, 
    they should follow exactly the same order and of same length.
    Inputs:
        pos_file_before: full filename for positions file before translation
        pos_file_after: full filename for positions file after translation
        save: whether save rotation and translation info, bool (default: True)
        save_folder: where to save rotation and translation info, None or string (default: same folder as pos_file_before)
        save_filename: filename specified to save rotation and translation points
        verbose: say something! bool (default: True)
    Outputs:
        R: rotation for positions, 2x2 array
        T: traslation of positions, array of 2
    Here's example for how to translate points
        translated_ps_before = np.dot(ps_before, R) + t
    """
    # load position_before
    if os.path.isfile(pos_file_before):
        ps_before = np.loadtxt(pos_file_before, delimiter=',')
    else:
        raise IOError(
            f"- Position file:{pos_file_before} file doesn't exist, exit!")
    # load position_after
    if os.path.isfile(pos_file_after):
        ps_after = np.loadtxt(pos_file_after, delimiter=',')
    else:
        raise IOError(
            f"- Position file:{pos_file_after} file doesn't exist, exit!")
    # do SVD decomposition to get best fit for rigid-translation
    c_before = np.mean(ps_before, axis=0)
    c_after = np.mean(ps_after, axis=0)
    H = np.dot((ps_before - c_before).T, (ps_after - c_after))
    U, _, V = np.linalg.svd(H)  # do SVD
    # calcluate rotation
    R = np.dot(V, U.T).T
    if np.linalg.det(R) < 0:
        R[:, -1] = -1 * R[:, -1]
    # calculate translation
    t = - np.dot(c_before, R) + c_after
    # here's example for how to translate points
    # translated_ps_before = np.dot(ps_before, R) + t
    if verbose:
        print(
            f"- Manually picked points aligned, rotation:\n{R},\n translation:{t}")
    
    return R, t

def correct_image3D_by_microscope_param(image3D:np.ndarray, microscope_params:dict):
    """Correct 3D image with microscopy parameter"""
    _image = image3D.copy()
    if not isinstance(microscope_params, dict):
        raise TypeError(f"Wrong inputt ype for microscope_params, should be a dict")
    # transpose
    if 'transpose' in microscope_params and microscope_params['transpose']:
        _image = _image.transpose((0,2,1))
    if 'flip_horizontal' in microscope_params and microscope_params['flip_horizontal']:
        _image = np.flip(_image, 2)
    if  'flip_vertical' in microscope_params and microscope_params['flip_vertical']:
        _image = np.flip(_image, 1)
    return _image

# Function to calculate translation for alignment of re-mounted samples
def calculate_translation(reference_im:np.ndarray, 
                          target_im:np.ndarray,
                          ref_to_tar_rotation:np.ndarray=None,
                          use_autocorr:bool=True,
                          alignment_kwargs:dict={},
                          verbose:bool=True,
                          ):
    """Calculate translation between two images with rotation """
    from math import pi
    import cv2
    ## quality check
    # images
    if np.shape(reference_im) != np.shape(target_im):
        raise IndexError(f"two images should be of the same shape")
    # rotation matrix
    if ref_to_tar_rotation is None:
        ref_to_tar_rotation = np.diag([1,1])
    elif np.shape(ref_to_tar_rotation) != tuple([2,2]):
        raise IndexError(f"wrong shape for rotation matrix, should be 2x2. ")
    # get dimensions
    _dz,_dx,_dy = np.shape(reference_im)
    # calculate angle
    if verbose:
        print(f"-- start calculating drift with rotation between images")
    _rotation_angle = np.arcsin(ref_to_tar_rotation[0,1])/pi*180
    _temp_new_rotation_M = cv2.getRotationMatrix2D((_dx/2, _dy/2), _rotation_angle, 1) # temporary rotation angle
    # rotate image
    if _rotation_angle != 0:
        _rot_target_im = np.array([cv2.warpAffine(_lyr, _temp_new_rotation_M, 
                                                  _lyr.shape, borderMode=cv2.BORDER_DEFAULT) 
                                   for _lyr in target_im], dtype=reference_im.dtype)
    else:
        _rot_target_im = target_im
    # calculate drift    
    _drift, _drift_flag = align_image(
        _rot_target_im,
        reference_im,
        precision_fold=10)

    if verbose:
        print(f"--- drift: {np.round(_drift,2)} pixels")
        
    return _rot_target_im, ref_to_tar_rotation, _drift, _drift_flag

def warp_3d_image(image, drift, chromatic_profile=None, 
                  warp_order=1, border_mode='constant', 
                  verbose=False):
    """Warp image given chromatic profile and drift"""
    _start_time = time.time()
    # 1. get coordiates to be mapped
    single_im_size = np.array(image.shape)
    _coords = np.meshgrid( np.arange(single_im_size[0]), 
            np.arange(single_im_size[1]), 
            np.arange(single_im_size[2]), )
    _coords = np.stack(_coords).transpose((0, 2, 1, 3)) # transpose is necessary 
    # 2. calculate corrected coordinates if chormatic abbrev.
    if chromatic_profile is not None:
        _coords = _coords + chromatic_profile 
    # 3. apply drift if necessary
    _drift = np.array(drift)
    if _drift.any():
        _coords = _coords - _drift[:, np.newaxis,np.newaxis,np.newaxis]
    # 4. map coordinates
    _corr_im = map_coordinates(image, 
                               _coords.reshape(_coords.shape[0], -1),
                               order=warp_order,
                               mode=border_mode, cval=np.min(image))
    _corr_im = _corr_im.reshape(np.shape(image)).astype(image.dtype)
    if verbose:
        print(f"-- finish warp image in {time.time()-_start_time:.3f}s. ")
    return _corr_im


def translate_segmentation(dapi_before, dapi_after, before_to_after_rotation, RNA_segment,
                           return_new_dapi=False,
                           verbose=True,
                           ):
    """ """
    
    # calculate drift
    _rot_dapi_after, _rot, _dft, _drift_flag = calculate_translation(dapi_before, dapi_after, before_to_after_rotation,)
    # get dimensions
    _dz,_dx,_dy = np.shape(dapi_before)
    _rotation_angle = np.arcsin(_rot[0,1])/math.pi*180
    
    if RNA_segment is not None:
        _seg_labels = np.array(RNA_segment)
        _rotation_angle = -1 * _rotation_angle
        _dft = -1 * _dft
    else:
        ValueError('RNA segmentation should be given!')
    # generate rotation matrix in cv2
    if verbose:
        print('- generate rotation matrix')
    _rotation_M = cv2.getRotationMatrix2D((_dx/2, _dy/2), _rotation_angle, 1)
    # rotate segmentation
    if verbose:
        print('- rotate segmentation label with rotation matrix')
    _rot_seg_labels = np.array(
        [cv2.warpAffine(_seg_layer,
                        _rotation_M, 
                        _seg_layer.shape, 
                        flags=cv2.INTER_NEAREST,
                        #borderMode=cv2.BORDER_CONSTANT,
                        borderMode=cv2.BORDER_REPLICATE,
                        #borderValue=int(np.min(_seg_labels)
                        )
            for _seg_layer in _seg_labels]
        )
    # warp the segmentation label by drift
    _dft_rot_seg_labels = warp_3d_image(_rot_seg_labels, _dft, 
        warp_order=0, border_mode='nearest')
    if return_new_dapi:
        return _dft_rot_seg_labels, _rot_dapi_after
    else:
        return _dft_rot_seg_labels, _dft, _drift_flag