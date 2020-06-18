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

raw_train = []
mem_train = []
dmv_train = []
sv_train = []
new_sv_labels_train = []
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
	sv_train.append( f['data'][:,:,:]. astype('uint64')) #(np.float32)

	print(name, '\n', np.unique(sv_train[-1]))
	print(len(np.unique(sv_train[-1])))
	print(np.max(sv_train[-1]))

	new_labels_train, maxlabel, mapping = vigra.analysis.relabelConsecutive(sv_train[-1])

	print(np.unique(new_labels_train))
	print(len(np.unique(new_labels_train)))
	print(np.max(new_labels_train), maxlabel)
	
	new_sv_labels_train.append(new_labels_train)

#do edge training 
rf = elf_workflow.edge_training(raw = raw_train, boundaries = mem_train, labels = dmv_train, use_2dws = False, watershed = new_sv_labels_train)

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
	sv_test = f['data'][:,:,:].astype('uint64')   #(np.float32)
	
	print(name, '\n', np.unique(sv_test))
	print(len(np.unique(sv_test))) 
	print(np.max(sv_test))

	new_labels, maxlabel, mapping = vigra.analysis.relabelConsecutive(sv_test)

	print(np.unique(new_labels))
	print(len(np.unique(new_labels))) 
	print(np.max(new_labels), maxlabel)


	#ground truth 
	ground_truth = np.zeros_like(mem_test, dtype = 'float32')
	ground_truth[128:384,  128:384, 128:384] = dmv_test
        #ground_truth.append(ground_truth_1)

	#run segmentation 
	segmentation = elf_workflow.multicut_segmentation(raw = raw_test, boundaries = mem_test, rf = rf, use_2dws = False, watershed = new_labels, multicut_solver = 'blockwise-multicut', solver_kwargs = {'internal_solver':'kernighan-lin', 'block_shape':[128,128,128]}, n_threads = 1) #multicut_solver = 'kernighan-lin')

	#save segmentation to h5 file 
	f = open_file('/scratch/gross/src/segmentation/results/' + name + '_mc_blockwise.h5', 'w')
	f.create_dataset('data', data = segmentation, compression = "gzip")
  	
	print(name, ' segmentation is done') 
