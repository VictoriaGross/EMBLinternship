import os
import numpy as np
import napari
import elf
from elf.io import open_file
import h5py
import vigra

filename_raw = '/g/schwab/Viktoriia/src/source/raw_crop.h5'

res_path = '/g/schwab/Viktoriia/src/source/upscaling_results/'  # + s4a2_mc_z_y_x_shape.h5

with open_file(filename_raw, 'r') as f:
	raw = f['data'][:, :, :].astype(np.float32)
	shape = f['data'].shape

print("Full  shape:", shape)

nz, ny, nx = shape

result_file = h5py.File(res_path + 'result_stitching_file_by_row.h5', mode='r')
result_dataset = result_file['data']


with napari.gui_qt():
	viewer = napari.Viewer()
	viewer.add_image(raw, name='raw')
	viewer.add_labels(result_dataset[:,:,:], name='stitched')

