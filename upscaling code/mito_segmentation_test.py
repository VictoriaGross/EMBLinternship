import os
import pickle
import numpy as np
import napari
import elf
import elf.segmentation.workflows as elf_workflow
from elf.io import open_file
import h5py
import vigra


################################################
# training part
###############################################

path_train = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/segmentation_results/'
rf_save_path = './rf_mito.pkl'


def train():
    print("start edge training")
    raw_train = []
    mem_train = []
    dmv_train = []
    sv_train = []
    new_sv_labels_train = []

    for name in ['s4a2_t016', 's4a2_t022', 's4a2_t028', 's4a2_t029']:
        # raw data
        filename = name + '/raw_crop_center256_256_256.h5'
        f = open_file(path_train + filename, 'r')
        raw_train.append(f['data'][:].astype(np.float32))

        # membrane prediction -- boundaries
        filename = name + '/mem_crop_center256_256_256.h5'
        f = open_file(path_train + filename, 'r')
        mem_train.append(f['data'][:].astype(np.float32))

        # ground truth for DMVs
        filename = name + '/results/raw_MITO.h5'
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

data_path_1 = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/20-04-23_S4_area2_Sam/'
names = ['s4a2_t010', 's4a2_t014']
for name in names:
    # raw data
    filename = name + '/raw.h5'
    f = open_file(data_path_1 + filename, 'r')
    raw_test = f['data'][:,:,:].astype(np.float32)

    # membrane prediction -- boundaries
    filename = name + '/mem.h5'
    f = open_file(data_path_1 + filename, 'r')
    mem_test = f['data'][:,:,:].astype(np.float32)

    # supervoxels
    filename = name + '/sv.h5'
    f = open_file(data_path_1 + filename, 'r')
    sv_test = f['data'][:, :, :].astype('uint64')  # (np.float32)
    new_labels, maxlabel, mapping = vigra.analysis.relabelConsecutive(sv_test)

    # run blockwise segmentation
    segmentation = elf_workflow.multicut_segmentation(raw=raw_test, boundaries=mem_test, rf=rf, use_2dws=False,
                                                      #watershed=new_labels,
                                                      multicut_solver='kernighan-lin', n_threads = 4)
    """
    segmentation = elf_workflow.multicut_segmentation(raw=raw_test, boundaries=mem_test, rf=rf, use_2dws=False,
                                                      watershed=new_labels, multicut_solver='blockwise-multicut',
                                                      solver_kwargs={'internal_solver': 'kernighan-lin',
                                                                     'block_shape': [128, 128, 128]},
                                                      n_threads=4)  # multicut_solver = 'kernighan-lin')
    """
    # save segmentation to h5 file
    f = open_file('./' + name + '_mc_512.h5', 'w')
    f.create_dataset('data', data=segmentation, compression="gzip")

    print(name, ' segmentation is done')
    """
    # run segmentation
    segmentation = elf_workflow.multicut_segmentation(raw=raw_test, boundaries=mem_test, rf=rf, use_2dws=False,
                                                      watershed=new_labels, multicut_solver='kernighan-lin')

    # save segmentation to h5 file
    f = open_file('/scratch/gross/src/segmentation/results/' + name + '_mc_non_blockwise.h5', 'w')
    f.create_dataset('data', data=segmentation, compression="gzip")

    print(name, ' segmentation is done')
    """
