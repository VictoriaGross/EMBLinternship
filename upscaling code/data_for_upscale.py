import numpy as np 

import napari 

import elf

from elf.io import open_file
import h5py 

filename1 = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/alignments/20-04-23_S4_area2_Sam/amst_inv_clip_black_bg.h5'
filename2 = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/alignments/20-04-25_S5_mock_cell1_2_Phoebe/amst_inv_black_bg.h5'

f = open_file(filename1, 'r') 
data1 = f['t00000/s00/0/cells']
print(data1.shape)

f = open_file(filename2, 'r') 
data2 = f['t00000/s00/0/cells']
print(data2.shape)

with napari.gui_qt(): 
	viewer = napari.Viewer()
	viewer.add_image(data1, name = 's4a2') 
