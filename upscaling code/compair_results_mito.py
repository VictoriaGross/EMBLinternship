import os
import numpy as np
import elf
from elf.io import open_file
import h5py
import vigra
from concurrent.futures import ThreadPoolExecutor
import sys
import pandas as pd
from pybdv import make_bdv
from pybdv import convert_to_bdv

filename_raw = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/alignments/20-04-23_S4_area2_Sam/amst_inv_clip_black_bg.h5'
with open_file(filename_raw, 'r') as f:
    full_shape = f['/t00000/s00/0/cells'].shape
res_path = '/g/schwab/Viktoriia/src/source/upscaling_results/whole_dataset_mito/updated_beta_0.6/'
res_path_j = '/g/schwab/hennies/project_corona/segmentation/upscale/20-04-23_S4_area2_Sam/seg_mito_scale1.0_2/'
res_filename_j = 'run_200617_00_mito_full_dataset.h5'
mask_filename = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/segmentation/MIBProcessing/s4a2_target_cell_mask.h5' #dataset data

iou_results = pd.DataFrame(data = [], columns = ['block', 'size in mc_elf', 'size in mc_j', 'iou'])

scale_factors = [[2,2,2], [2, 2, 2], [4, 4, 4]]
mode = 'nearest'

def get_iou(start_index = (0, 0, 0)):
    # open file
    z, y, x = start_index
    filename = 's4a2_mc_' + str(z) + '_' + str(y) + '_' + str(x) + '_' + '1024_1024_1024.h5'
    filename_bdv = 's4a2_mc_' + str(z) + '_' + str(y) + '_' + str(x) + '_bin_4.h5'
    if not os.path.exists(res_path + filename_bdv):
        with h5py.File(res_path + filename, 'r') as f:
            current_label = f['data'][:,:,:].astype('uint64')
        current_label, max_label, mapping = vigra.analysis.relabelConsecutive(current_label, start_label=1, keep_zeros=True)
        binary_current_label = (current_label != 1).astype('uint')
        make_bdv(binary_current_label, res_path + filename_bdv, downscale_factors=scale_factors, downscale_mode=mode)
        #convert_to_bdv(res_path + filename, 'data', res_path + filename_bdv, downscale_factors = scale_factors, downscale_mode = mode)
    with h5py.File(res_path + filename_bdv, 'r') as f:
        bin_down_label = f['/t00000/s00/2/cells'][:,:,:]
    shape = bin_down_label.shape

    bb = np.s_[z // 4 : z // 4 + shape[0], y // 4 : y // 4 + shape[1], x // 4 : x // 4 + shape[2]]
    with h5py.File(mask_filename, 'r') as f:
        mask = f['data'][bb]
        bin_down_label = bin_down_label * mask

    with h5py.File(res_path_j + res_filename_j, 'r') as f:
        control_results = f['bin_4'][bb] * mask

    size1 = np.count_nonzero(bin_down_label)
    size2 = np.count_nonzero(control_results)
    interception = bin_down_label * control_results
    size_inter = np.count_nonzero(interception)
    if size1 == 0 and size2 == 0:
        iou = 1
    else:
        iou = size_inter / (size2 + size1 - size_inter)
    print(start_index, size1, size2, iou)
    return (start_index, size1, size2, iou)

with h5py.File(res_path_j + res_filename_j, 'r') as f:
    print(f.keys())
    print(f['bin_4'].shape)
    print(f['scale_1'].shape)
    print(f['scale_2'].shape)
    print(f['scale_4'].shape)


# nz, ny,nx = full_shape
# step_z, step_y, step_x = 768, 768, 768
# x_positions = np.arange(0, nx, step_x)
# y_positions = np.arange(0, ny, step_y)
# z_positions = np.arange(0, nz, step_z)
# n = len(x_positions) * len(y_positions) * len(z_positions)
# variables = []
# for z in z_positions:
#     for y in y_positions:
#         for x in x_positions:
#             variables.append((z, y, x))
#
# n_workers = 4
# with ThreadPoolExecutor(max_workers=n_workers) as tpe:
#     # submit the tasks
#     tasks = [
#         tpe.submit(get_iou, start_index = (z, y, x))
#         for z, y, x in variables
#     ]
#     # get results
#     results = [task.result() for task in tasks]
# i = 0
# for res in results:
#     iou_results.loc[i] = list(res)
#     i += 1
#
# print(iou_results)
# iou_results.to_csv(res_path + 'iou_results_mito_by_blocks_1.csv', sep = '\t')