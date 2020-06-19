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

raw = []
membrane_prediction = []
dmv = []
supervoxels = []
multicut = []

ground_truth = [] 
for name in ['s4a2_t010', 's4a2_t012']: 
	#raw data
	data_path = '/scratch/emcf/s4a2_mc/'
	filename = name +  '_raw.h5'
	print(data_path + filename)
	f = open_file(data_path + filename, 'r')
	raw.append(f['data'][:,:,:].astype(np.float32))

	#membrane prediction -- boundaries 
	filename =  name + '_mem.h5'
	f = open_file(data_path + filename, 'r')
	membrane_prediction.append( f['data'][:,:,:].astype(np.float32))

	# ground truth for DMVs 
	filename = name + '_dmv.h5'
	f = open_file(data_path + filename, 'r')
	dmv.append( f['data'] [:,:,].astype(np.float32))

	#supervoxels 
	filename = name + '_sv.h5'
	f = open_file(data_path + filename, 'r')
	supervoxels.append( f['data'][:,:,:].astype(np.float32))

	#results of multicut segmentation 
	filename = name + '_mc.h5'
	f = open_file(data_path + filename, 'r') 
	multicut.append(f['data'][:,:,:].astype(np.float32))

	ground_truth_1 = np.zeros_like(membrane_prediction[-1], dtype = 'float32')
	ground_truth_1[128:384,  128:384, 128:384] = dmv[-1]
	ground_truth.append(ground_truth_1)


with napari.gui_qt():
	viewer = napari.Viewer()
	viewer.add_image(raw[0], name = 'raw')
	viewer.add_image(membrane_prediction[0], name = 'membrane_prediction')
	viewer.add_labels(multicut[0], name = 'multicut')
	viewer.add_image(ground_truth[0], name = 'DMV')
	viewer.add_image(supervoxels[0], name = 'supervoxels')

	viewer1 = napari.Viewer()
	viewer1.add_image(raw[1], name = 'raw')
	viewer1.add_image(membrane_prediction[1], name = "membrane_prediction")
	viewer1.add_labels(multicut[1], name = 'multicut')
	viewer1.add_image(ground_truth[1], name = 'DMV')
	viewer1.add_image(supervoxels[1], name = 'supervoxels')


