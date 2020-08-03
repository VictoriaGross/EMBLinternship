import numpy as np

import napari

import elf
from elf.io import open_file
import h5py

import vigra

path = '/g/schwab/Viktoriia/src/source/'
res_path = '/g/schwab/Viktoriia/src/source/upscaling_results/whole_dataset_dmv/'  # + s4a2_mc_z_y_x_shape.h5
filename_raw = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/alignments/20-04-23_S4_area2_Sam/amst_inv_clip_black_bg.h5'

#with open_file(path + 'sv_crop.h5') as f:
#	sv = f['data'][700:,:,:]


with open_file(filename_raw, 'r') as f:
	raw = f['/t00000/s00/0/cells'][768:1024 + 768,768:1024 + 768,1536:1536 + 1024].astype(np.float32)

print(np.min(raw))
print(np.max(raw))
print(raw.size)
print(np.sum(raw))
print(np.count_nonzero(raw != 0.))

#with open_file(res_path + 's4a2_mc_0_0_512_1024_768_768.h5', 'r') as f:
#	labels1 = f['data'][:, :, :]
#labels1, max_label, mapping = vigra.analysis.relabelConsecutive(labels1.astype('uint64'), start_label= 1, keep_zeros=True)
#labels1 = vigra.analysis.applyMapping(labels=labels1, mapping={1 : 0}, allow_incomplete_mapping=True)


with open_file(res_path + 's4a2_mc_768_768_1536_1024_1024_1024.h5', 'r') as f:
	labels2 = f['data'][:, :, :].astype('uint32')
	print('done')
#labels2, max_label, mapping = vigra.analysis.relabelConsecutive(labels2.astype('uint64'), keep_zeros=True)
#labels2 = vigra.analysis.applyMapping(labels=labels2, mapping={max_label + 1 : 0}, allow_incomplete_mapping=True)
#labels_2 = np.zeros((1024, 1280, 1792))
#labels_2[:, 1024:, :] = labels2

"""
with open_file(path + 'upscaling_results/s4a2_mc_blockwise.h5', 'r') as f: 
	segmentation = f['data'][:,:, :]
with open_file(path + 'upscaling_results/s4a2_mc_blockwise_1.h5', 'r') as f: 
	segmentation_1 = f['data'][:,:,:] 
with open_file(path + 'upscaling_results/s4a2_mc_blockwise_2.h5', 'r') as f: 
	segmentation_2 = f['data'][:, :, :]
#with open_file(path + 'upscaling_results/s4a2_mc_blockwise_3.h5', 'r') as f: 
with open_file(path + 'res.h5', 'r') as f:
	segmentation_3 = f['data'][:, :, :]
	


nz, ny, nx = raw.shape 
labels = np.zeros((nz, ny, nx))
max_label = 0


old_maxlabel = max_label 
segmentation, max_label, mapping = vigra.analysis.relabelConsecutive(segmentation, start_label = max_label + 1, keep_zeros = True) 
labels[:, ny // 2 :, nx // 2:] = vigra.analysis.applyMapping(labels = segmentation, mapping = {old_maxlabel + 1 : 0}, allow_incomplete_mapping = True) 
old_maxlabel = max_label 
segmentation_1, max_label, mapping = vigra.analysis.relabelConsecutive(segmentation_1, start_label = max_label + 1, keep_zeros = True ) 
labels[:, 1024 :, :1024] = vigra.analysis.applyMapping(labels = segmentation_1, mapping = {old_maxlabel + 1 : 0}, allow_incomplete_mapping = True) 
old_maxlabel = max_label 
segmentation_2, max_label, mapping = vigra.analysis.relabelConsecutive(segmentation_2, start_label = max_label + 1, keep_zeros = True ) 
labels[:, : 1024, :1024] = vigra.analysis.applyMapping(labels = segmentation_2, mapping = {old_maxlabel + 1 : 0}, allow_incomplete_mapping = True) 
old_maxlabel = max_label  
segmentation_3, max_label, mapping = vigra.analysis.relabelConsecutive(segmentation_3, start_label = max_label + 1, keep_zeros = True)
labels[:, : ny //2 + 128, 1024:] = vigra.analysis.applyMapping(labels = segmentation_3, mapping = {old_maxlabel + 1 : 0}, allow_incomplete_mapping = True) 
old_maxlabel = max_label  
"""
#with open_file('./concatenated_results.h5', 'w') as f: 
#	f.create_dataset('data', data = labels.astype('float32'), compression = 'gzip') 

#labels_1 = np.zeros((nz, ny, nx)) 
#labels_1[:, 1024:, :1024] = segmentation_1 
#labels_2 = np.zeros((nz, ny, nx)) 
#labels_2 [:, :1024, : 1024] = segmentation_2 
#labels_3 = np.zeros((nz, ny, nx)) 
#labels_3 [:, :ny//2 + 128, 1024:] = segmentation_3 


with napari.gui_qt(): 
	viewer = napari.Viewer()
	#viewer.add_image(sv)
	viewer.add_image(raw, name = 'raw')
	#viewer.add_labels(labels1, name = '1')
	viewer.add_labels(labels2, name = '2')
	#viewer.add_labels(labels_1, name = 'mc2') 
	#viewer.add_labels(labels_2, name = 'mc3')
	#viewer.add_labels(labels_3, name = 'mc4')
