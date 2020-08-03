import os
import pickle
import numpy as np
import napari
import elf
import elf.segmentation.workflows as elf_workflow
from elf.io import open_file
import h5py
import vigra
import threading
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

#filename_raw = '/g/schwab/Viktoriia/src/source/raw_crop.h5'
filename_raw = '/g/emcf/common/5792_Sars-Cov-2/Exp_070420/FIB-SEM/alignments/20-04-23_S4_area2_Sam/amst_inv_clip_black_bg.h5'

#res_path = '/g/schwab/Viktoriia/src/source/upscaling_results/'  # + s4a2_mc_z_y_x_shape.h5
res_path = '/g/schwab/Viktoriia/src/source/upscaling_results/whole_dataset_dmv/'  # + s4a2_mc_z_y_x_shape.h5

with open_file(filename_raw, 'r') as f:
	#raw = f['/t00000/s00/0/cells'][:, :, :].astype(np.float32)
	shape = f['/t00000/s00/0/cells'].shape

print("Full  shape:", shape)

nz, ny, nx = shape

with h5py.File(res_path + 'result_stitching_file.h5', mode = 'w') as f:
	f.create_dataset('data', shape = shape, compression = 'gzip', dtype = 'uint64')

result_file = h5py.File(res_path + 'result_stitching_file.h5', mode='a')
result_dataset = result_file['data']

#a = np.zeros(shape)
#result_dataset.write_direct(a, np.s_[:,:,:], np.s_[:,:,:])
maxlabel = 0

def stitching_row (start_index = (0,0,0), row_count = 0, step = 512, shape = (1024, 2304, 1792), maxlabel = maxlabel, axis = 2, use_stitched = False):
	name = 'data_' + 'row' * (axis == 2) + 'column' * (axis == 1) + 'z-layer' * (axis == 0) + str(row_count)

	result_file.create_dataset(name, shape = shape, compression = 'gzip', dtype = 'uint64')
	result_dataset_row = result_file[name]

	z = start_index[0]
	y = start_index[1]
	x = start_index[2]
	coordinate = list(start_index)
	end = False

	size_cube = step + 256
	current_shape = (0, 0, 0)
	while not end:
		bb = np.s_[z: min(z + size_cube, nz), y: min(y + size_cube, ny), x: min(x + size_cube, nx)]
		print(z, y, x)
		print(bb)
		filename_results = 's4a2_mc_' + str(z) + '_' + str(y) + '_' + str(x) + '_' + str(size_cube) + '_' + str(size_cube) + '_' + str(size_cube) + '.h5'
		f = open_file(res_path + filename_results, 'r')
		current_label = f['data'][:, :, :]
		if current_shape == (0,0,0):
			current_shape = current_label.shape
			part = np.s_[: current_shape[0], : current_shape[1], : current_shape[2]]
			current_label, new_maxlabel, mapping = vigra.analysis.relabelConsecutive(current_label.astype('uint64'), start_label=maxlabel + 1,
															keep_zeros=True)
			current_label = vigra.analysis.applyMapping(current_label, mapping={maxlabel + 1: 0}, allow_incomplete_mapping=True)
			maxlabel = new_maxlabel
			result_dataset_row.write_direct(current_label, np.s_[:, :, :],
											np.s_[: current_shape[0], : current_shape[1], : current_shape[2]])
		else:
			maxlabel, current_shape = stitching_function(result_dataset_row, current_label, current_shape, current_maxlabel=maxlabel, start_index_2=(0, 0, x))
		print('row stitching function', current_shape, maxlabel)
		coordinate[axis] += step
		z, y, x = coordinate
		end = coordinate[axis] >= shape[axis]
	return current_shape, maxlabel

def stitching_column(step_x = 768, step_y = 768, z = 0, maxlabel = 0, column_index = 0, total_shape = (1024, 2304, 1792), max_row_index = 0, n_workers = 1):
	row_index = max_row_index
	shapes_of_rows = list()
	maxlabels = []
	if n_workers == 1:
		for y in range(0, ny, step_y):
			row_shape, maxlabel = stitching_row(start_index=(z, y, 0), row_count=row_index, step=step_x,
												shape=total_shape, maxlabel=maxlabel, axis=2)
			row_index += 1
			shapes_of_rows.append(row_shape)
			print('column stitching, row_index, maxlabel, y, z ', row_index, maxlabel, y, z)
	else:
		y_positions = np.arange(0, ny, step_y)
		row_indexes = np.arange(row_index, row_index + len(y_positions))
		variables = []
		for i in range(len(y_positions)):
			variables.append((y_positions[i], row_indexes[i]))
		with ThreadPoolExecutor(max_workers = n_workers) as tpe:
			#submit the tasks
			tasks = [
				tpe.submit(stitching_row, start_index=(z, y, 0), row_count=row_index, step=step_x,
												shape=total_shape, maxlabel=maxlabel, axis=2)
				for y, row_index in variables
			]
			#get results
			results = [task.result() for task in tasks]
	print ('threads results ', results)
	for res in results:
		shapes_of_rows.append(res[0])
		maxlabels.append(res[1])
	maxlabel = maxlabels[0]
	#row_index = len(list(result_file.keys())) - 1
	y = 0
	x = 0
	current_shape = (0, 0, 0)

	name = 'data_column' + str(column_index)

	result_file.create_dataset(name, shape=total_shape, compression='gzip', dtype='uint64')
	result_dataset_column = result_file[name]

	for i in range (row_index - max_row_index):
		print(y)
		name = 'data_row' + str(i + max_row_index)
		current_label = result_file[name][:shapes_of_rows[i][0],:shapes_of_rows[i][1],:shapes_of_rows[i][2]]
		print(current_label.shape)
		if current_shape == (0, 0, 0):
			current_shape = current_label.shape
			part = np.s_[: current_shape[0], y : current_shape[1], : current_shape[2]]
			result_dataset_column.write_direct(current_label, np.s_[:, :, :], part)
		else:
			maxlabel, current_shape = stitching_function(result_dataset_column, current_label, current_shape, current_maxlabel=maxlabel,
													  start_index_2=(z, y, x))
		y += step_y
		print(current_shape, maxlabel)
	return current_shape, maxlabel, row_index + 1

def stitch_total(step_x = 768, step_y = 768, step_z = 768, maxlabel = 0, total_shape = (1024, 2304, 1792), n_workers = 1):
	print('start stitch total')
	row_index = 0
	column_index = 0
	shapes_of_columns = list()
	# threads = list()
	for z in range(0, nz, step_z):
		print('z = ', z)

		column_shape, maxlabel, row_index = stitching_column(step_x = step_x, step_y = step_y, z = z, maxlabel = maxlabel,
															 column_index = column_index, total_shape = total_shape, max_row_index = row_index, n_workers = n_workers)

		column_index += 1
		shapes_of_columns.append(column_shape)
		print('column_shape = ', column_shape)

	# row_index = len(list(result_file.keys())) - 1
	z = 0
	y = 0
	x = 0
	current_shape = (0, 0, 0)
	#result_dataset = result_file['data']
	print('column index ', column_index)
	for i in range(column_index):
		name = 'data_column' + str(i)
		current_label = result_file[name][:shapes_of_columns[i][0], : shapes_of_columns[i][1], :shapes_of_columns[i][2]]
		if current_shape == (0, 0, 0):
			current_shape = current_label.shape
			part = np.s_[: current_shape[0], y: current_shape[1], : current_shape[2]]
			result_dataset.write_direct(current_label, np.s_[:, :, :], part)
		else:
			maxlabel, current_shape = stitching_function(result_dataset, current_label, current_shape,
														 current_maxlabel=maxlabel,
														 start_index_2=(z, y, x))
		z += step_z
		print(current_shape, maxlabel)
	print('done')
	return current_shape, maxlabel

def stitching_function(result_dataset, new_region, current_shape, current_maxlabel=0, start_index_2 = None):

	print('start stitching')

	new_region, max_label, mapping = vigra.analysis.relabelConsecutive(new_region.astype('uint64'), start_label=current_maxlabel + 1, keep_zeros=True)
	new_region = vigra.analysis.applyMapping(labels=new_region, mapping={current_maxlabel + 1: 0}, allow_incomplete_mapping=True)

	current_maxlabel = max_label
	print('new max label', max_label)

	shape2 = new_region.shape
	print ('shape of new region', shape2)

	result_data = np.zeros(shape2)
	index = np.s_[start_index_2[0] : start_index_2[0] + shape2[0],
			start_index_2[1] : start_index_2[1] + shape2[1],
			start_index_2[2] : start_index_2[2] + shape2[2]]
	result_dataset.read_direct(result_data, index, np.s_[:,:,:])
	print('done reading')

	# get shapes
	shape1 = current_shape
	shape2 = new_region.shape

	shape = np.zeros(3).astype('int')
	for i in range(3):
		shape[i] = max(start_index_2[i] + shape2[i], shape1[i])

	shape = tuple(shape)

	print('new shape', shape)

	# non-overlapping regions
	print('start non-overlap')
	for i in range(3):
		if start_index_2[i] + shape2[i] > shape1[i]:
			start = [0, 0, 0]
			start[i] = shape1[i]
			part = np.s_[start[0] : start_index_2[0] + shape2[0], start[1] : start_index_2[1] + shape2[1], start[2] : start_index_2[2] + shape2[2]]
			start[i] = shape1[i] - start_index_2[i]
			part1 = np.s_[start[0] : start_index_2[0] + shape2[0], start[1] : start_index_2[1] + shape2[1], start[2] : start_index_2[2] + shape2[2]]
			#result_dataset.write_direct(new_region, source_sel = part1, dest_sel = part)
			print(part1)
			result_data[part1] = np.copy(new_region[part1])
	print('done with non-overlapping')

	# overlapping region
	start_in1 = np.array(start_index_2)
	end_in1 = np.array((min(start_in1[0] + shape2[0], shape1[0]),
							min(start_in1[1] + shape2[1], shape1[1]),
							min(start_in1[2] + shape2[2], shape1[2])))

	part_in1 = np.s_[start_in1[0]: end_in1[0], start_in1[1]: end_in1[1], start_in1[2]: end_in1[2]]

	start_in2 = np.array((0,0,0))
	end_in2 = np.array((min(shape2[0], shape1[0] - start_in1[0]),
							min(shape2[1], shape1[1] - start_in1[1]),
							min(shape2[2], shape1[2] - start_in1[2])))
	part_in2 = np.s_[start_in2[0]: end_in2[0], start_in2[1]: end_in2[1], start_in2[2]: end_in2[2]]

	part_in1 = part_in2

	start_new = np.array(start_index_2)
	end_new = np.array((min(start_new[0] + shape2[0], shape1[0]),
							min(start_new[1] + shape2[1], shape1[1]),
							min(start_new[2] + shape2[2], shape1[2])))
	part_new = np.s_[start_new[0]: end_new[0], start_new[1]: end_new[1], start_new[2]: end_new[2]]

	labels_overlap1 = np.unique(result_data[part_in1])
	labels_overlap2 = np.unique(new_region[part_in2])

	print('labels_overlap_1', labels_overlap1)
	print('labels overlap 2', labels_overlap2)

	if labels_overlap2.size == 1 and labels_overlap2 == [0.] and labels_overlap1.size == 1 and labels_overlap1 == [0.] :
		result_dataset.write_direct(result_data, np.s_[:, :, :], index)
		print('no overlap, done stitching ')
		return (current_maxlabel + 1, shape)



	print ('part in 1', part_in1)
	print ('part in 2', part_in2)
	print ('part in overall ', part_new)

	mapping_overlap = {}
	sizes_overlap1 = {}
	sizes_overlap2 = {}
	sizes_intercept = {}

	for i in labels_overlap1:
		if i != 0:
			sizes_overlap1[i] = np.count_nonzero(result_data[part_in1] == i)

	for i in labels_overlap2:
		if i != 0:
			sizes_overlap2[i] = np.count_nonzero(new_region[part_in2] == i)

	print('done with sizes of each object')

	shape_overlap = result_data[part_in1].shape

	for z in range(shape_overlap[0]):
		for y in range(shape_overlap[1]):
			for x in range(shape_overlap[2]):
				l1 = result_data[part_in1][z, y, x]
				l2 = new_region[part_in2][z, y, x]
				if l1 != 0 and l2 != 0:
					if l2 in mapping_overlap.keys() and not (l1 in mapping_overlap[l2]):
						np.append(mapping_overlap[l2], l1)
						np.append(sizes_intercept[l2], 1)
					elif l2 in mapping_overlap.keys() and l1 in mapping_overlap[l2]:
						sizes_intercept[l2][np.where(mapping_overlap[l2] == l1)[0][0]] += 1
					elif not (l2 in mapping_overlap.keys()):
						mapping_overlap[l2] = np.array([l1])
						sizes_intercept[l2] = np.array([1])
	#print(mapping_overlap)
	print('done mapping')

	iou = {}
	for l2 in mapping_overlap.keys():
		n = mapping_overlap[l2].size
		iou[l2] = np.zeros(n)
		for i in range(n):
			l1 = mapping_overlap[l2][i]
			iou[l2][i] = sizes_intercept[l2][i] / (sizes_overlap1[l1] + sizes_overlap2[l2] - sizes_intercept[l2][i])
	print('done iou')

	for z in range(shape_overlap[0]):
		for y in range(shape_overlap[1]):
			for x in range(shape_overlap[2]):
				l1 = result_data[part_in1][z, y, x]
				l2 = new_region[part_in2][z, y, x]
				if l1 == 0 and l2 != 0:
					if l2 in mapping_overlap.keys():
						n = mapping_overlap[l2].size
						i = 0
						if n > 1:
							i = np.argmax(iou[l2])
						if iou[l2][i] >= 0.8:
							result_data[part_in2][z, y, x] = mapping_overlap[l2][i]
						else:
							result_data[part_in2][z, y, x] = l2
					else:
						result_data[part_in2][z, y, x] = l2
	print('done overlap')

	for l2 in mapping_overlap.keys():
		n = mapping_overlap[l2].size
		i = 0
		if n > 1:
			i = np.argmax(iou[l2])
		if iou[l2][i] >= 0.8:
			result_data = np.where(result_data == l2, mapping_overlap[l2][i], result_data)
	print('done stitching')
	result_dataset.write_direct(result_data, np.s_[:, :, :], index)
	return (current_maxlabel + 1, shape)

def stitch(labels1, labels2, overlap=(0, 256, 256), current_maxlabel=0, axis=2, labels=None, start_index_1=None,
		   start_index_2=None):
	print('start stitching')

	# relabel to consecutive and to having background as 0
	labels1, max_label, mapping = vigra.analysis.relabelConsecutive(labels1.astype('uint64'),
																	start_label=current_maxlabel + 1, keep_zeros=True)
	labels1 = vigra.analysis.applyMapping(labels=labels1, mapping={current_maxlabel + 1: 0},
										  allow_incomplete_mapping=True)

	current_maxlabel = max_label
	print(max_label)

	labels2, max_label, mapping = vigra.analysis.relabelConsecutive(labels2.astype('uint64'),
																	start_label=current_maxlabel + 1, keep_zeros=True)
	labels2 = vigra.analysis.applyMapping(labels=labels2, mapping={current_maxlabel + 1: 0},
										  allow_incomplete_mapping=True)

	current_maxlabel = max_label
	print(max_label)

	""" 
	For now we assume that the overlap is exactly 256 pixels and it is only along x and y axes.
	We assume that labels1 correspond to the "left" one, and labels2 correspond to the "right" one. 
	"""
	# get shapes
	shape1 = labels1.shape
	shape2 = labels2.shape

	if not start_index_2:
		shape = list(max(shape1, shape2))
		shape[axis] = shape1[axis] + shape2[axis] - overlap[axis]
		shape = tuple(shape)
	else:
		if not start_index_1:
			start_index_1 = (0, 0, 0)
		shape = np.zeros(3).astype('int')
		for i in range(3):
			if start_index_1[i] < start_index_2[i]:
				shape[i] = max(start_index_2[i] - start_index_1[i] + shape2[i], shape1[i])
			elif start_index_1[i] > start_index_2[i]:
				shape[i] = max(start_index_1[i] - start_index_2[i] + shape1[i], shape2[i])
			else:
				shape[i] = max(shape1[i], shape2[i])
		shape = tuple(shape)

	new_labels = np.zeros(shape)
	print('shape', shape)

	# non-overlapping regions

	if not start_index_2:
		end1 = list(shape1)
		end1[axis] = shape1[axis] - overlap[axis]
		start2 = [0, 0, 0]
		start2[axis] = shape1[axis]

		part1 = np.s_[: end1[0], : end1[1], : end1[2]]
		part2 = np.s_[start2[0]:, start2[1]:, start2[2]:]
		start_label2 = [0, 0, 0]
		start_label2[axis] = overlap[axis]
		part_label2 = np.s_[start_label2[0]:, start_label2[1]:, start_label2[2]:]

		new_labels[part1] = np.copy(labels1[part1])
		new_labels[part2] = np.copy(labels2[part_label2])
	else:
		if not start_index_1:
			start_index_1 = (0, 0, 0)

		start1 = np.array(start_index_1) - np.array(start_index_2)
		for i in range(start1.size):
			if start1[i] < 0:
				start1[i] = 0
		end1 = list(shape1)

		start2 = np.array(start_index_2) - np.array(start_index_1)
		for i in range(start2.size):
			if start2[i] < 0:
				start2[i] = 0
		end2 = start2 + np.array(shape2)

		part1 = np.s_[start1[0]: end1[0], start1[1]: end1[1], start1[2]: end1[2]]
		part2 = np.s_[start2[0]: end2[0], start2[1]: end2[1], start2[2]: end2[2]]

		new_labels[part1] = np.copy(labels1)
		new_labels[part2] = np.copy(labels2)

	print('done with non-overlapping')

	# overlapping region
	if not start_index_2:
		start_in1 = [0, 0, 0]
		start_in1[axis] = shape1[axis] - overlap[axis]
		end_in1 = list(shape1)
		part_in1 = np.s_[start_in1[0]: end_in1[0], start_in1[1]: end_in1[1], start_in1[2]: end_in1[2]]

		start_in2 = [0, 0, 0]
		end_in2 = list(shape2)
		end_in2[axis] = overlap[axis]
		part_in2 = np.s_[start_in2[0]: end_in2[0], start_in2[1]: end_in2[1], start_in2[2]: end_in2[2]]

		start_new = [0, 0, 0]
		start_new[axis] = shape1[axis] - overlap[axis]
		end_new = list(shape)
		end_new[axis] = shape1[axis]
		part_new = np.s_[start_new[0]: end_new[0], start_new[1]: end_new[1], start_new[2]: end_new[2]]
	else:
		if not start_index_1:
			start_index_1 = (0, 0, 0)
		start_in1 = np.array((max(start_index_2[0] - start_index_1[0], 0),
							  max(start_index_2[1] - start_index_1[1], 0),
							  max(start_index_2[2] - start_index_1[2], 0)))

		end_in1 = np.array((min(start_in1[0] + shape2[0], shape1[0]),
							min(start_in1[1] + shape2[1], shape1[1]),
							min(start_in1[2] + shape2[2], shape1[2])))

		part_in1 = np.s_[start_in1[0]: end_in1[0], start_in1[1]: end_in1[1], start_in1[2]: end_in1[2]]

		start_in2 = np.array((max(start_index_1[0] - start_index_2[0], 0),
							  max(start_index_1[1] - start_index_2[1], 0),
							  max(start_index_1[2] - start_index_2[2], 0)))

		end_in2 = np.array((min(shape2[0], shape1[0] - start_in1[0]),
							min(shape2[1], shape1[1] - start_in1[1]),
							min(shape2[2], shape1[2] - start_in1[2])))

		part_in2 = np.s_[start_in2[0]: end_in2[0], start_in2[1]: end_in2[1], start_in2[2]: end_in2[2]]

		start_new = np.array((max(max(start_index_1[0], start_index_2[0]) - min(start_index_1[0], start_index_2[0]), 0),
							  max(max(start_index_1[1], start_index_2[1]) - min(start_index_1[1], start_index_2[1]), 0),
							  max(max(start_index_1[2], start_index_2[2]) - min(start_index_1[2], start_index_2[2]),
								  0)))

		end_new = np.array((min(start_new[0] + shape2[0], shape1[0]),
							min(start_new[1] + shape2[1], shape1[1]),
							min(start_new[2] + shape2[2], shape1[2])))

		part_new = np.s_[start_new[0]: end_new[0], start_new[1]: end_new[1], start_new[2]: end_new[2]]
	"""
	#average/combine binary images 
	binary1 = labels1[part_in1] != 0 
	binary2 = labels2[part_in2] != 0
	
	binary1 = binary1.astype(np.uint32) 
	binary2 = binary2.astype(np.uint32) 
	binary = np.maximum(binary1, binary2) 

	print('done binary') 
	
	#map averaged binary image to classes 
	labels1_wo_bckg = vigra.analysis.applyMapping(labels = labels1[part_in1].astype('uint64'), mapping = {0 : current_maxlabel +1 }, allow_incomplete_mapping = True) 

	new_labels[part_new] = np.multiply(binary, labels1_wo_bckg) 

	print('start connected components') 
	new_labels = vigra.analysis.labelMultiArrayWithBackground(new_labels.astype('uint32'), 26) 
	print('done connected components') 
	"""

	labels_overlap1 = np.unique(labels1[part_in1])
	labels_overlap2 = np.unique(labels2[part_in2])

	mapping_overlap = {}
	sizes_overlap1 = {}
	sizes_overlap2 = {}
	sizes_intercept = {}

	for i in labels_overlap1:
		if i != 0:
			sizes_overlap1[i] = np.count_nonzero(labels1[part_in1] == i)

	for i in labels_overlap2:
		if i != 0:
			sizes_overlap2[i] = np.count_nonzero(labels2[part_in2] == i)

	shape_overlap = labels1[part_in1].shape
	for z in range(shape_overlap[0]):
		for y in range(shape_overlap[1]):
			for x in range(shape_overlap[2]):
				l1 = labels1[part_in1][z, y, x]
				l2 = labels2[part_in2][z, y, x]
				if l1 != 0 and l2 != 0:
					if l2 in mapping_overlap.keys() and not (l1 in mapping_overlap[l2]):
						np.append(mapping_overlap[l2], l1)
						np.append(sizes_intercept[l2], 1)
					elif l2 in mapping_overlap.keys() and l1 in mapping_overlap[l2]:
						sizes_intercept[l2][np.where(mapping_overlap[l2] == l1)[0][0]] += 1
					elif not (l2 in mapping_overlap.keys()):
						mapping_overlap[l2] = np.array([l1])
						sizes_intercept[l2] = np.array([1])
					# print('interception', sizes_intercept)
	iou = {}
	for l2 in mapping_overlap.keys():
		n = mapping_overlap[l2].size
		iou[l2] = np.zeros(n)
		for i in range(n):
			l1 = mapping_overlap[l2][i]
			iou[l2][i] = sizes_intercept[l2][i] / (sizes_overlap1[l1] + sizes_overlap2[l2] - sizes_intercept[l2][i])
	# print( 'iou', iou)
	for z in range(shape_overlap[0]):
		for y in range(shape_overlap[1]):
			for x in range(shape_overlap[2]):
				l1 = labels1[part_in1][z, y, x]
				l2 = labels2[part_in2][z, y, x]
				if l1 != 0:
					new_labels[part_new][z, y, x] = l1
				elif l2 != 0:
					if l2 in mapping_overlap.keys():
						i = np.argmax(iou[l2])
						if iou[l2][i] >= 0.8:
							new_labels[part_new][z, y, x] = mapping_overlap[l2][i]
						else:
							new_labels[part_new][z, y, x] = l2
					else:
						new_labels[part_new][z, y, x] = l2

	for l2 in mapping_overlap.keys():
		n = mapping_overlap[l2].size
		i = 0
		if n > 1:
			i = np.argmax(iou[l2])
		# print(i, iou[l2][i], l2, mapping_overlap[l2][i])
		if iou[l2][i] >= 0.8:
			new_labels = np.where(new_labels == l2, mapping_overlap[l2][i], new_labels)
	print('done stitching')
	return (new_labels, current_maxlabel + 1)


"""
count = 1
row_index = 0
z = 0

shapes_of_rows = list()
threads = list()
for y in range (0, ny, 768):
	#add threading
	#x = threading.Thread(target= stitching_row, args = ( start_index=(z, y, 0), row_count= row_index, step=768, shape=(1024, 2304, 1792)))
	row_shape, maxlabel = stitching_row(start_index=(z, y, 0), row_count= row_index, step=768, shape=(1024, 2304, 1792), maxlabel = maxlabel, axis = 2)
	#x.start()
	row_index += 1
	shapes_of_rows.append(row_shape)
	print(row_index, maxlabel, y, z)

#row_index = len(list(result_file.keys())) - 1
y = 0
x = 0
z = 0
current_shape = (0, 0, 0)

print('start stitching rows')
print('there are ', row_index, 'rows')
print(result_dataset.shape)

#row_shape, maxlabel = stitching_row(start_index=(0, 0, 0), row_count=row_index, step=768, shape=(1024, 2304, 1792),
#									maxlabel=maxlabel, axis=1)


for i in range (row_index):
	print(y)
	name = 'data_row' + str(i)
	current_label = result_file[name][:shapes_of_rows[i][0],:shapes_of_rows[i][1],:shapes_of_rows[i][2]]
	print(current_label.shape)
	if current_shape == (0, 0, 0):
		current_shape = current_label.shape
		part = np.s_[: current_shape[0], y : current_shape[1], : current_shape[2]]
		result_dataset.write_direct(current_label, np.s_[:, :, :], part)
	else:
		maxlabel, current_shape = stitching_function(result_dataset, current_label, current_shape, current_maxlabel=maxlabel,
												  start_index_2=(z, y, x))
	y += 768
	print(current_shape, maxlabel)

	
		for x in range (0, nx, 768):
			if (x + y + z > 0) and (nx - x > 256 and ny - y > 256 and nz - z > 256): 
				bb = np.s_[z : min(z + 1024, nz), y : min(y + 1024, ny), x : min(x + 1024, nx)]
				print(z, y, x)
				print(bb)
				filename_results = 's4a2_mc_' + str(z) + '_' + str(y) + '_' + str(x) + '_1024_1024_1024.h5'
				f = open_file(res_path + filename_results, 'r') 
				current_label = f['data'][:, :, :] 
				max_label, current_shape = stitching_function(result_dataset, current_shape, current_maxlabel = max_label, start_index_2 = (z, y, x))
				print(current_shape, max_label)
				count += 1
	
"""
shape = list(shape)
shape[0] = 2536
shape = tuple(shape)
current_shape, maxlabel = stitch_total(step_x = 768, step_y = 768, step_z = 768, maxlabel = 0, total_shape = shape, n_workers = 4)

"""
with napari.gui_qt():
	viewer = napari.Viewer()
	#viewer.add_image(raw, name='raw')
	viewer.add_labels(result_dataset[:,:,:], name='stitched')
"""
