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

#raw data
data_path = '/scratch/emcf/s4a2_mc/' 
filename = 's4a2_t012_raw.h5' 
f = open_file(data_path + filename, 'r')
raw = f['data'][:,:,:].astype(np.float32)

#membrane prediction -- boundaries 
filename = 's4a2_t012_mem.h5'
f = open_file(data_path + filename, 'r')
membrane_prediction = f['data'][:,:,:].astype(np.float32)

# ground truth for DMVs 
filename = 's4a2_t012_dmv.h5'
f = open_file(data_path + filename, 'r') 
dmv = f['data'] [:,:,].astype(np.float32)

#supervoxels 
filename = 's4a2_t012_sv.h5'
f = open_file(data_path + filename, 'r') 
supervoxels = f['data'][:,:,:].astype(np.float32)


ground_truth = np.zeros_like(membrane_prediction, dtype = 'float32')
ground_truth[128:384,  128:384, 128:384] = dmv

with napari.gui_qt():
	viewer = napari.Viewer()
	viewer.add_image(raw, name = 'raw')
	viewer.add_image(membrane_prediction, name = "membrane_prediction")
	viewer.add_image(ground_truth, name = 'DMV')
	viewer.add_image(supervoxels, name = 'supervoxels')

#shape of ground truth 
nx, ny, nz = dmv.shape
#reshape raw and boundaries for training 
#because of the error on edge-training try to use a block 100, 100, 100
raw_train = raw[128:384, 128:384, 128:384] #.astype(np.float32)
membrane_prediction_train = membrane_prediction[128:384, 128:384, 128:384] #.astype(np.float32)
#dmv_train = dmv.astype(np.float32)
#sv_train = supervoxels.astype(np.float32)


#do edge training 
rf = elf_workflow.edge_training(raw = raw_train, boundaries = membrane_prediction_train, labels = dmv, use_2dws = False, watershed = supervoxels) 

print("edge training is done")

#raw_segment = raw.astype(np.float32) 
#membrane_predict_segment = (membrane_prediction).astype(np.float32)

#try blockwise segmentation on the same file (but on the entire one, not only the center cube) 
segmentation = elf_workflow.multicut_segmentation(raw = raw, boundaries = membrane_prediction, rf = rf, use_2dws = False, multicut_solver = 'blockwise-multicut', solver_kwargs = {'internal_solver':'kernighan-lin', 'block_shape':[100,100,100]}, n_threads = 1)

#save segmentation to h5 file 
f = h5py.File('/scratch/emcf/s4a2_mc/s4a2_t012_mc.h5', 'w') 
f.create_dataset('data', data = segmentation) 

with napari.gui_qt(): 
	viewer = napari.Viewer() 
	viewer.add_image(raw, name = 'raw')
	viewer.add_image(membrane_prediction, name = 'membrane_prediction') 
	viewer.add_labels(segmentation, name = 'multicut') 
	viewer.add_image(ground_truth, name = 'DMV')
 

#try out with other file 

#raw data
filename = 's4a2_t010_raw.h5'
f = open_file(data_path + filename, 'r')
raw = f['data'][:,:,:].astype(np.float32)

#membrane prediction -- boundaries 
filename = 's4a2_t010_mem.h5'
f = open_file(data_path + filename, 'r')
membrane_prediction = f['data'][:,:,:].astype(np.float32)

# ground truth for DMVs 
filename = 's4a2_t010_dmv.h5'
f = open_file(data_path + filename, 'r')
dmv = f['data'] [:,:,].astype(np.float32)

#supervoxels 
filename = 's4a2_t010_sv.h5'
f = open_file(data_path + filename, 'r')
supervoxels = f['data'][:,:,:].astype(np.float32)


ground_truth = np.zeros_like(membrane_prediction, dtype = 'float32')
ground_truth[128:384,  128:384, 128:384] = dmv

segmentation = elf_workflow.multicut_segmentation(raw = raw, boundaries = membrane_prediction, rf = rf, use_2dws = False, multicut_solver = 'blockwise-multicut', solver_kwargs = {'internal_solver':'kernighan-lin', 'block_shape':[100,100,100]}, n_threads = 1)

#save segmentation to h5 file 
f = h5py.File('/scratch/emcf/s4a2_mc/s4a2_t010_mc.h5', 'w')
f.create_dataset('data', data = segmentation)

with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw, name = 'raw')
        viewer.add_image(membrane_prediction, name = 'membrane_prediction')
        viewer.add_labels(segmentation, name = 'multicut')
        viewer.add_image(ground_truth, name = 'DMV')

#segmentation worked auite nicely, but there are maybe some parts over-segmented 
