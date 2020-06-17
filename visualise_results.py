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

data_path_2 = '/scratch/emcf/segmentation_results/'
raw_test = []
mem_test = []
dmv_test = []
sv_test = []

ground_truth = []
multicut = []

names = ['s4a2_t002', 's4a2_t018'] #, 's4a2_t028']

data_path_1 = '/scratch/emcf/segmentation_inputs/'
data_path_results = '/scratch/gross/src/segmentation/results/'

for name in names: #['s4a2_t002', 's4a2_t018']: # 's4a2_t024', 's4a2_t028']:
        #raw data
	filename = name +  '/raw.h5'
	f = open_file(data_path_1 + filename, 'r')
	raw_test.append( f['data'][:,:,:].astype(np.float32))

        #membrane prediction -- boundaries 
	filename =  name + '/mem.h5'
	f = open_file(data_path_1 + filename, 'r')
	mem_test.append(f['data'][:,:,:].astype(np.float32))

        # ground truth for DMVs 
	filename = name + '/results/raw_DMV.h5'
	f = open_file(data_path_2 + filename, 'r')
	dmv_test.append(f['data'] [:,:,].astype(np.float32))

        #supervoxels 
	filename = name + '/sv.h5'
	f = open_file(data_path_1 + filename, 'r')
	sv_test.append( f['data'][:,:,:].astype(np.float32))

        #ground truth 
	ground_truth_1 = np.zeros_like(mem_test[-1], dtype = 'float32')
	ground_truth_1[128:384,  128:384, 128:384] = dmv_test[-1]
	ground_truth.append(ground_truth_1)

	#multicut 
	filename = name + '_mc.h5'
	f = open_file(data_path_results + filename, 'r') 
	multicut.append(f['data'][:,:,:].astype(np.float32))

n =  len(names)
viewer = []
with napari.gui_qt():
	for i in range(n): 
	        viewer.append( napari.Viewer())
        	viewer[-1].add_image(raw_test[i], name = names[i] + 'raw')
	        viewer[-1].add_image(mem_test[i], name = 'membrane_prediction')
	        viewer[-1].add_labels(multicut[i], name = 'multicut')
	        viewer[-1].add_image(ground_truth[i], name = 'DMV')
        #viewer.add_image(supervoxels[0], name = 'supervoxels')

        #viewer1 = napari.Viewer()
        #viewer1.add_image(raw_test[1], name = 'raw')
        #viewer1.add_image(mem_test[1], name = "membrane_prediction")
        #viewer1.add_labels(multicut[1], name = 'multicut')
        #viewer1.add_image(ground_truth[1], name = 'DMV')
        #viewer1.add_image(supervoxels[1], name = 'supervoxels')



