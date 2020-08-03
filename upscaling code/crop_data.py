import numpy as np

import napari

import elf
import elf.segmentation as segm
import elf.segmentation.workflows as elf_workflow
import elf.segmentation.multicut as elf_multicut
import elf.segmentation.features as elf_feats
import elf.segmentation.watershed as elf_ws

from elf.io import open_file
import h5py

import vigra

path_sv = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/upscale/20-04-23_S4_area2_Sam/sv_scale1/'

filename_mem = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/upscale/20-04-23_S4_area2_Sam/mem_pred_scale1.h5'

filename_raw = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/alignments/20-04-23_S4_area2_Sam/amst_inv_clip_black_bg.h5'


start_z, start_y, start_x = 1408, 384, 1152#1280, 256, 1024
end_z, end_y, end_x = 2560 - 128, 2816 - 128, 3072 - 128 

f = open_file(filename_raw, 'r')  
raw_crop = f['t00000/s00/0/cells'][start_z : end_z, start_y : end_y, start_x : end_x].astype(np.float32) 

f = open_file('/g/schwab/Viktoriia/src/source/raw_crop.h5', 'w') 
f.create_dataset('data', data = raw_crop, compression = 'gzip') 

f = open_file(filename_mem, 'r') 
print(f['data'].shape, f['data'])
print(f['compute_map'].shape)
mem_crop = f['data'][start_z : end_z, start_y : end_y, start_x : end_x]#.astype(np.float32)
print(mem_crop.shape)

f = open_file('/g/schwab/Viktoriia/src/source/mem_crop.h5', 'w') 
f.create_dataset('data', data = mem_crop, compression = 'gzip') 

with napari.gui_qt(): 
	viewer = napari.Viewer() 
	viewer.add_image(raw_crop)
	viewer.add_image(mem_crop)

