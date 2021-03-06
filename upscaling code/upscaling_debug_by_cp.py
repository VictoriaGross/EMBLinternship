import os
import pickle
import numpy as np

import elf.segmentation.workflows as elf_workflow
from elf.io import open_file

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
# segmentation part

filename_raw = '/g/schwab/Viktoriia/src/source/raw_crop.h5'
filename_mem = '/g/schwab/Viktoriia/src/source/mem_crop.h5'
filename_sv = '/g/schwab/Viktoriia/src/source/sv_crop.h5'

with open_file(filename_raw, 'r') as f:
    shape = f['data'].shape
print("Full  shape:", shape)


# Run the full segmentation pipeline to debug Viktoriia's
run_full = True

# run the full segmentation to reproduce the issue
if run_full:
    nz, ny, nx = shape
    mc_blocks = [256, 256, 256]
    #bb = np.s_[:, : ny//2 + 128,  1024:]
    bb = np.s_[:,:,:]
# run on a smaller cutout
else:
    bb = np.s_[:256, :256, :256]
    mc_blocks = [128, 128, 128]
    bb = np.s_[:]

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
# save segmentation to h5 file
# res_path = '/g/schwab/Viktoriia/src/source/upscaling_results/' + 's4a2_mc_blockwise_3.h5'
res_path = './res_full.h5'
with open_file(res_path, 'w') as f:
    f.create_dataset('data', data=segmentation, compression="gzip")

