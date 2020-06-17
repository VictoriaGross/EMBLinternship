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

raw_train = []
mem_train = []
dmv_train = []
sv_train = []
#get data for training
data_path = '/scratch/emcf/segmentation_results/' 
for name in ['s4a2_t010', 's4a2_t012']:
        #raw data
        filename = name +  '/raw_crop_center256_256_256.h5'
        f = open_file(data_path + filename, 'r')
        raw_train.append(f['data'][:,:,:].astype(np.float32))

        #membrane prediction -- boundaries 
        filename =  name + '/mem_crop_center256_256_256.h5'
        f = open_file(data_path + filename, 'r')
        mem_train.append( f['data'][:,:,:].astype(np.float32))

        # ground truth for DMVs 
        filename = name + '/results/raw_DMV.h5'
        f = open_file(data_path + filename, 'r')
        dmv_train.append( f['data'] [:,:,].astype(np.float32))

        #supervoxels 
        filename = name + '/sv_crop_center256_256_256.h5'
        f = open_file(data_path + filename, 'r')
        sv_train.append( f['data'][:,:,:].astype(np.float32))

#do edge training 
rf = elf_workflow.edge_training(raw = raw_train, boundaries = mem_train, labels = dmv_train, use_2dws = False, watershed = sv_train)

print("edge training is done")

# get validation data 
#raw_test = []
#mem_test = []
#dmv_test = []
#sv_test = []
#ground_truth = []
#get data for training
data_path_1 = '/scratch/emcf/segmentation_inputs/'
for name in ['s4a2_t002', 's4a2_t018',  's4a2_t028']:
	#raw data
	filename = name +  '/raw.h5'
	f = open_file(data_path_1 + filename, 'r')
	raw_test = f['data'][:,:,:].astype(np.float32)

	#membrane prediction -- boundaries 
	filename =  name + '/mem.h5'
	f = open_file(data_path_1 + filename, 'r')
	mem_test = f['data'][:,:,:].astype(np.float32)

	# ground truth for DMVs 
	filename = name + '/results/raw_DMV.h5'
	f = open_file(data_path + filename, 'r')
	dmv_test = f['data'] [:,:,:].astype(np.float32)

	#supervoxels 
	filename = name + '/sv.h5'
	f = open_file(data_path_1 + filename, 'r')
	sv_test = f['data'][:,:,:].astype(np.float32)
	
	#ground truth 
	ground_truth = np.zeros_like(mem_test, dtype = 'float32')
	ground_truth[128:384,  128:384, 128:384] = dmv_test
        #ground_truth.append(ground_truth_1)

	#run segmentation 
	segmentation = elf_workflow.multicut_segmentation(raw = raw_test, boundaries = mem_test, rf = rf, use_2dws = False, watershed = sv_test,  multicut_solver = 'blockwise-multicut', solver_kwargs = {'internal_solver':'kernighan-lin', 'block_shape':[64,64,64]}, n_threads = 1)

	#save segmentation to h5 file 
	f = open_file('/scratch/gross/src/segmentation/results/' + name + '_mc.h5', 'w')
	f.create_dataset('data', data = segmentation, compression = "gzip")
  	
	print(name, ' segmentation is done') 
