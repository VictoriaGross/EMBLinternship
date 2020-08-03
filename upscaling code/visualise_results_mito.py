import numpy as np

import napari

import elf
from elf.io import open_file
import h5py

import vigra

data_path_results = '/g/schwab/Viktoriia/src/source/'
data_path_1 = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/20-04-23_S4_area2_Sam/'
names = ['s4a2_t010', 's4a2_t014']
raw = []
multicut_512 = []
multicut_256 = []
for name in names:
	filename = name +  '/raw.h5'
	f = open_file(data_path_1 + filename, 'r')
	raw.append( f['data'][:,:,:].astype(np.float32))
	shape = raw[-1].shape

	if name == 's4a2_t010':
		filename = name + '_mc_512.h5'
		f = open_file(data_path_results + filename, 'r')
		multicut_512.append(f['data'][:,:,:].astype('uint32'))
	else:
		multicut_512.append(np.zeros(shape))

	filename = name + '_mc_with_sv.h5'
	f = open_file(data_path_results + filename, 'r')
	multicut_256.append(np.zeros(shape))
	multicut_256[-1][128:384, 128:384, 128:384]= f['data'][:, :, :].astype('uint32')

n = len(names)
viewer = []
with napari.gui_qt():
	for i in range(n):
		viewer.append( napari.Viewer())
		viewer[-1].add_image(raw[i], name = names[i] + 'raw')
		viewer[-1].add_labels(multicut_512[i], name='multicut1')
		viewer[-1].add_labels(multicut_256[i], name = 'multicut2')
