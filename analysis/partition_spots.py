import os
import numpy as np
import pandas as pd
import h5py

# Partition spots into segmented cells

def is_index_out_of_bounds(indices, array_shape):
    # Use numpy.any to check if any index is out of bounds
    boundary_check = False
    for i, upper_bound in enumerate(array_shape):
        if np.any(np.logical_or(np.less(indices[i], 0), np.greater_equal(indices[i], upper_bound))):
            boundary_check = True
    return boundary_check

def partition_spots(fov, spot_folder, feature_folder, filtered_segment_folder, output_folder, 
                    total_bits = 194, pixel_xy = 108, pixel_z = 250, imageSize = (50,2048,2048), overwrite=False):

    # define file names
    # TODO: make it more flexible system
    spot_file = os.path.join(spot_folder, 'Conv_zscan_'+str(fov).zfill(3)+'.hdf5')
    feature_file = os.path.join(feature_folder, 'feature_data_'+str(fov)+'.hdf5')
    filtered_segment_file = os.path.join(filtered_segment_folder, 'Conv_zscan_'+str(fov).zfill(3)+'.npy')
    output_file = os.path.join(output_folder, 'Conv_zscan_'+str(fov).zfill(3)+'_CandSpots.csv')
    # return if file already exists and not overwrite
    if not overwrite and os.path.exists(output_file):
        return
    
    # return if no spot file is found
    if not os.path.exists(spot_file):
        print(f'FOV {fov} has NOT been processed. Skip...')
        return

    # get bit to channel information
    color_usage = pd.read_csv(os.path.join(spot_folder, 'Color_Usage.csv'), index_col=0)
    bit_to_channel = {}
    for i, row in color_usage.iterrows():
        for color in color_usage.columns:
            if pd.notna(row[color]):
                bit_to_channel[int(row[color].split('c')[-1])] = color
    
    # get spot information
    spots_each_bit = {}
    with h5py.File(spot_file) as file:
        for _group in file.keys():
            if ('c' in _group) and ('_' not in _group):
                _bit = int(_group.split('c')[-1])
                spots_each_bit[_bit] = np.array(file[_group]['spots'])

    # Do not process this file when the spot fitting is not complete
    if len(spots_each_bit)!=total_bits:
        return

    # get segmentation and cell ID uid conversion
    cell_id_to_uid = {}
    with h5py.File(feature_file) as file:
        for uid in file['featuredata']:
            cell_id_to_uid[file['featuredata'][uid].attrs['label']] = str(uid)
        segment = np.array(file['dapi_label']['label3D'])
    
    # initialize dictionary
    columns = ['height', 'z', 'x', 'y', 'background', 'sigma_z', 'sigma_x', 'sigma_y', 'sin_t', 'sin_p', 'eps']
    
    dict_spot = {}
    dict_spot['fov_id'] = []
    dict_spot['cell_id'] = []
    dict_spot['uid'] = []
    dict_spot['bit'] = []
    for col in columns:
        dict_spot[col] = []

    # define important parameters
    filtered_segment = np.load(filtered_segment_file)
    kept_cell_ids = np.unique(filtered_segment)
    kept_cell_ids = kept_cell_ids[kept_cell_ids!=0]

    for bit, spot_list in spots_each_bit.items():
        for _spot in spot_list:
            _coord = np.floor(_spot[1:4]).astype(int)
            cell_id = None
            # if the fitted spots are out of the boundary
            if is_index_out_of_bounds(_coord, imageSize):
                continue
            # when the fitted spot is in the segment
            if segment[tuple(_coord.T)]!=0:
                cell_id = segment[tuple(_coord.T)]
            else:
                z_min, z_max = max(0, int(_coord[0]-(2/(pixel_z/1000)))), min(imageSize[0]-1, int(_coord[0]+(2/(pixel_z/1000))))
                x_min, x_max = max(0, int(_coord[1]-(3/(pixel_xy/1000)))), min(imageSize[1]-1, int(_coord[1]+(3/(pixel_xy/1000))))
                y_min, y_max = max(0, int(_coord[2]-(3/(pixel_xy/1000)))), min(imageSize[2]-1, int(_coord[2]+(3/(pixel_xy/1000))))
                _local_seg = segment[z_min:z_max, x_min:x_max, y_min:y_max]
                val, count = np.unique(_local_seg, return_counts=True)
                _id_count = count[val!=0]
                _ids = val[val!=0]
                if len(_ids)>0:
                    cell_id = _ids[np.argmax(_id_count)]

            if (cell_id is not None) and (cell_id in kept_cell_ids):
                uid = cell_id_to_uid[cell_id]
                dict_spot['fov_id'].append(fov)
                dict_spot['cell_id'].append(cell_id)
                dict_spot['uid'].append(uid)
                dict_spot['bit'].append(bit)
                for i, col in enumerate(columns):
                    dict_spot[col].append(_spot[i])
    
    df_spot = pd.DataFrame(dict_spot)
    df_spot['pixel_x'] = pixel_xy
    df_spot['pixel_y'] = pixel_xy
    df_spot['pixel_z'] = pixel_z
    df_spot['channel'] = df_spot['bit'].apply(lambda x: bit_to_channel[x])

    # filter candidate spots
    df_spot = df_spot[df_spot['sigma_x']<=3].copy()
    df_spot = df_spot[df_spot['sigma_y']<=3].copy()
    df_spot = df_spot[df_spot['sigma_z']<=3].copy()

    df_spot.to_csv(output_file, index=False)