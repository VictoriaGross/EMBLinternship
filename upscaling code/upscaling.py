import os
import pickle
import numpy as np
import napari
import elf
import elf.segmentation.workflows as elf_workflow
from elf.io import open_file
import h5py
import vigra


# training part
path_train = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/segmentation_results/'
rf_save_path = './rf.pkl'


def train():
    print("start edge training")
    raw_train = []
    mem_train = []
    dmv_train = []
    sv_train = []
    new_sv_labels_train = []

    for name in ['s4a2_t010', 's4a2_t012']:
        # raw data
        filename = name + '/raw_crop_center256_256_256.h5'
        f = open_file(path_train + filename, 'r')
        raw_train.append(f['data'][:].astype(np.float32))

        # membrane prediction -- boundaries
        filename = name + '/mem_crop_center256_256_256.h5'
        f = open_file(path_train + filename, 'r')
        mem_train.append(f['data'][:].astype(np.float32))

        # ground truth for DMVs
        filename = name + '/results/raw_DMV.h5'
        f = open_file(path_train + filename, 'r')
        dmv_train.append(f['data'][:].astype(np.float32))

        # supervoxels
        filename = name + '/sv_crop_center256_256_256.h5'
        f = open_file(path_train + filename, 'r')
        sv_train.append(f['data'][:].astype('uint64'))  # (np.float32)

        newlabels_train, maxlabel, mapping = vigra.analysis.relabelConsecutive(sv_train[-1])
        new_sv_labels_train.append(newlabels_train)

    # do edge training
    rf = elf_workflow.edge_training(raw=raw_train, boundaries=mem_train, labels=dmv_train,
                                    use_2dws=False, watershed=new_sv_labels_train)
    print("edge training is done")
    return rf


if os.path.exists(rf_save_path):
    with open(rf_save_path, 'rb') as f:
        rf = pickle.load(f)
else:
    rf = train()
    with open(rf_save_path, 'wb') as f:
        pickle.dump(rf, f)


##################################################################
#segmentation part 


#path_sv = '/g/emcf/common/5702_Sars-cov-2/Exp_070420/FIB-SEM/segmentation/upscale/20-04-23_S4_area2_Sam/sv_scale1/'
#filename_mem = '/g/emcf/common/5702_Sars-cov-2/Exp_070420/FIB-SEM/segmentation/upscale/20-04-23_S4_area2_Sam/mem_pred_scale1.h5'
#filename_raw = '/g/emcf/common/5702_Sars-cov-2/Exp_070420/FIB-SEM/alignments/20-04-23_S4_area2_Sam/amst_inv_clip_black_bg.h5'

filename_raw = '/g/schwab/Viktoriia/src/source/raw_crop.h5'
filename_mem = '/g/schwab/Viktoriia/src/source/mem_crop.h5'
filename_sv = '/g/schwab/Viktoriia/src/source/sv_crop.h5'

res_path = '/g/schwab/Viktoriia/src/source/upscaling_results/' #+ s4a2_mc_z_y_x_shape.h5

with open_file(filename_raw, 'r') as f:
    shape = f['data'].shape
print("Full  shape:", shape)


nz, ny, nx = shape
mc_blocks = [256, 256, 256]
#bb = np.s_[:, : 1536, : 1536]
#bb = np.s_[:,:,:]

def mc_segmentation (bb, mc_blocks, filename_raw, filename_mem, filename_sv): 
	f = open_file(filename_raw, 'r')
	data_raw = f['data'][bb].astype(np.float32) 
	shape = data_raw.shape

	f = open_file(filename_mem, 'r') 
	data_mem = f['data'][bb].astype(np.float32).reshape(data_raw.shape)
	assert data_mem.shape == shape

	f = open_file(filename_sv, 'r')
	data_sv = f['data'][bb].astype('uint64')
	assert data_sv.shape == shape

	print("Final shape:", shape)

	# run blockwise segmentation
	print("Start segmentation")
	segmentation = elf_workflow.multicut_segmentation(raw=data_raw,
                                                  boundaries=data_mem,
                                                  rf=rf, use_2dws=False,
                                                  watershed=data_sv,
                                                  multicut_solver='blockwise-multicut',
                                                  solver_kwargs={'internal_solver': 'kernighan-lin',
                                                                 'block_shape': mc_blocks},
                                                  n_threads=8)  # multicut_solver = 'kernighan-lin')
	print('segmentation is done')
	return segmentation

#################
#run segmentations for all the cubes 

for z in range (0, nz, 1024):
	for y in range (0, ny, 512):
		for x in range (0, nx, 512):
			if (nx - x > 256 and ny - y > 256 and nz - z > 256): 
				bb = np.s_[z : min(z + 768, nz), y : min(y + 768, ny), x : min(x + 768, nx)]
				print(z, y, x)
				print(bb)
			
				#filename = s4a2_mc_z_y_x_shape.h5
				filename_results = 's4a2_mc_' + str(z) + '_' + str(y) + '_' + str(x) + '_1024_768_768.h5'
				if os.path.exists(res_path + filename_results):
					print('already done') 
				else:
					segmentation = mc_segmentation(bb, mc_blocks, filename_raw, filename_mem, filename_sv) 
					f = open_file(res_path + filename_results, 'w')
					f.create_dataset('data', data=segmentation, compression="gzip")
					print('done writing') 




