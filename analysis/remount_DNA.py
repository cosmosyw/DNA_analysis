# utilize to align remounted DNA to RNA

from ..image import dax
from ..preprocess import alignment
import h5py
import numpy as np
import os

RNA_channels = [750, 647, 488, 405]
DNA_channels = [750, 647, 561, 488, 405]
imageSize = (50, 2048, 2048)


def get_remounted_DNA_mask(rna_dax_file, dna_dax_file, rna_segment_file, 
                           rna_correction, dna_correction, microscope_dict,
                           rotation_matrix, output_folder, overwrite=False):
    
    # get output file name
    output_filename = os.path.basename(dna_dax_file).replace('.dax', '.npy')
    
    # save
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    output_file = os.path.join(output_folder, output_filename)
    
    if overwrite or (not os.path.exists(output_file)):
        # load dax file
        RNA_dapi = dax.Dax_Processor(rna_dax_file, RNA_channels, imageSize, rna_correction, microscope_dict)
        DNA_dapi = dax.Dax_Processor(dna_dax_file, DNA_channels, imageSize, dna_correction, microscope_dict)

        # load and correct image
        print('Loading RNA DAPI image')
        RNA_dapi.load_image()
        RNA_dapi.correct_image(['405'])
        rna_dapi = RNA_dapi.im_405

        print('Loading DNA DAPI image')
        DNA_dapi.load_image()
        DNA_dapi.correct_image(['405'])
        dna_dapi = DNA_dapi.im_405

        # release RAM
        del RNA_dapi
        del DNA_dapi

        with h5py.File(rna_segment_file, 'r') as file:
            print('Loading RNA segment')
            # Navigate to the 'dapi' group and get the 'label' dataset
            label_dataset = file['dapi_label']['label3D']  # Adjust group and dataset names as needed

            # Access the data from the dataset
            label_data = np.array(label_dataset)

        # generate DNA segment
        print('Start alignment')
        dna_segment, drift, _drift_flag = alignment.translate_segmentation(rna_dapi, dna_dapi, rotation_matrix, label_data)

        # save
        print('Save DNA segment')
        np.save(output_file, dna_segment)

        # drift output
        drift_output = os.path.join(output_folder, output_filename.replace('.npy', '.txt'))
        with open(drift_output, 'a') as f:
            f.write(str(drift)+'\n')
            f.write(_drift_flag)
        
    else:
        print(f'{output_file} already exists')

    return
