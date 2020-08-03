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

start_z, start_y, start_x = 1408, 384, 1152 #1280, 256, 1024
end_z, end_y, end_x = 2560 - 128, 2816 - 128, 3072 - 128


new_labels = np.zeros((end_z - start_z, end_y - start_y, end_x - start_x), dtype = 'uint32')
max_label = 0
#new_labels, maxlabel, mapping = vigra.analysis.relabelConsecutive(sv_test)

for z in range (start_z - 128, end_z - 384 + 1, 256):
	for y in range(start_y - 128, end_y - 384 + 1, 256):
		for x in range(start_x - 128, end_x - 384 + 1, 256):
			filename = str(z) + '_' + str(y) + '_' + str(x) + '.h5'
			f = open_file(path_sv + filename, 'r') 
			if len(list(f.keys())) != 0: 
				sv_current = f['data'][128:384, 128:384,128:384].astype('uint32')
				new_sv, new_max, mapping = vigra.analysis.relabelConsecutive(sv_current, start_label = max_label + 1) 
				max_label = new_max
				x0 = x - start_x + 128 
				y0 = y - start_y + 128 
				z0 = z - start_z + 128
				new_labels[z0:z0 + 256, y0:y0 + 256, x0:x0 + 256] = np.copy(new_sv) 


f = open_file('/g/schwab/Viktoriia/src/source/sv_crop_1.h5', 'w')
f.create_dataset('data', data = new_labels, compression = 'gzip')

with napari.gui_qt(): 
	viewer = napari.Viewer()
	viewer.add_image(new_labels, name = 'sv') 

