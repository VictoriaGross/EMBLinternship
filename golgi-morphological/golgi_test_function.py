import numpy as np 
import elf
import napari
from elf.io import open_file
import h5py
import vigra

def modify_probability(raw, probability): 
	dilated = vigra.filters.multiGrayscaleDilation(probability, 10000)
	thresholded_raw = raw <= 110
	modified_probability = np.multiply(dilated, thresholded_raw)
	return modified_probability

def golgi_filter_1(raw, probability): 
	#dilated = vigra.filters.multiGrayscaleDilation(probability, 10000)
	#thresholded_raw = raw <= 110
	#modified_probability = np.multiply(dilated, thresholded_raw)
	thresholded_probability_1 = probability >= (0.5 * np.max(probability)) #modified_probability >= 0.5
	opening = vigra.filters.multiBinaryOpening(thresholded_probability_1, 5) 
	#return opening 
	return thresholded_probability_1

def golgi_filter_2(raw, probability): 
	#dilated = vigra.filters.multiGrayscaleDilation(probability, 10000)
	#thresholded_raw = raw <= 110
	#modified_probability = np.multiply(dilated, thresholded_raw)

	thresholded_probability_1 = probability >= 0.55 # modified_probability >= 0.5
	
	connected_components_1 = vigra.analysis.labelMultiArrayWithBackground(thresholded_probability_1.astype('uint8'), 26)
	
	connected_erosion = vigra.filters.multiBinaryErosion(connected_components_1.astype('uint8'), 5)
	connected_opening_4 = vigra.filters.multiGrayscaleDilation(connected_erosion.astype(np.float32), 10)
	connected_close = vigra.filters.multiGrayscaleClosing(connected_components_1.astype(np.float32), 10)
	connected_close_1 = vigra.filters.multiBinaryClosing((connected_components_1  >= 1000).astype('uint8'), 10)
	connected_open = vigra.filters.multiGrayscaleOpening(connected_components_1.astype(np.float32), 15) #10)
	#connected_open_1 = vigra.filters.multiGrayscaleOpening(connected_components_1.astype(np.float32), 15)
	#return connected_close_1
	return connected_open


"""
path = '/g/schwab/Viktoriia/src/source/'
#import raw file 
with open_file(path + 'raw_crop.h5', 'r') as f:
        #first part 
        raw_1 = f['data'][400:650,1600:1900,600:900].astype(np.float32)
        #second part containing candidates 
        raw_2 = f['data'][500:650, 200:600, 300:700].astype(np.float32)

#import probability exported from ilastik 
#use the chanel 0 that correspond to golgi label 
with open_file (path + 'golgi_segmentation/golgi_probability_ilastik.h5', 'r') as f:
#with open_file (path + 'golgi-ilastik-uint8.h5') as f:
        probability_1 = f['exported_data'][400:650,1600:1900,600:900,0] #.reshape(raw.shape) #.astype('uint8')
        probability_2 = f['exported_data'][500:650,200:600,300:700,0]
"""

path = '/g/schwab/Viktoriia/src/source/golgi_segmentation/'
with h5py.File(path + 'golgi_1.h5', 'r') as f:
	raw_1 = f['data'][:,:,:] 
with h5py.File(path + 'golgi_2.h5', 'r') as f:
	raw_2 = f['data'][:,:,:] 
with h5py.File(path + 'golgi_3.h5', 'r') as f:
	raw_3 = f['data'][:,:,:] 
with h5py.File(path + 'golgi_4.h5', 'r') as f:
	raw_4 = f['data'][:,:,:] 

with h5py.File(path + 'golgi_1-data_Probabilities Stage 2.h5', 'r') as f:
        probability_1 = f['exported_data'][:,:,:, 0]
with h5py.File(path + 'golgi_2-data_Probabilities Stage 2.h5', 'r') as f:
        probability_2 = f['exported_data'][:,:,:, 0]
with h5py.File(path + 'golgi_3-data_Probabilities Stage 2.h5', 'r') as f:
        probability_3 = f['exported_data'][:,:,:, 0]
with h5py.File(path + 'golgi_4-data_Probabilities Stage 2.h5', 'r') as f:
        probability_4 = f['exported_data'][:,:,:, 0]


"""
output_1 = golgi_filter_1(raw_1, probability_1) 
output_2 = golgi_filter_1(raw_2, modify_probability(raw_2, probability_2))

output_3 = golgi_filter_2(raw_1, probability_1)
output_4 = golgi_filter_2(raw_2, probability_2) #modify_probability(raw_2, probability_2))
"""
viewer = [] 
n = 4
raw = [raw_1, raw_2, raw_3, raw_4]
probability = [probability_1, probability_2, probability_3, probability_4]

output = [] 
for i in range(n): 
	output.append(golgi_filter_1(raw[i], probability[i])) 
#output = [output_1, output_2, output_3, output_4]
with napari.gui_qt():
	for i in range(n): 
		viewer.append( napari.Viewer())
		viewer[-1].add_image(raw[i], name = 'raw') 
		viewer[-1].add_image(probability[i], name = 'golgi', colormap = 'red')
		viewer[-1].add_image(output[i], name = 'output', colormap = 'green') 
		#viewer[-1].add_image(output[i + n], name = 'output_1', colormap = 'blue') 
		for i in range (1, len(viewer[-1].layers)): 
			viewer[-1].layers[i].opacity = 0.3



#test for the whole dataset 
"""
with h5py.File(path + 'raw_crop.h5', 'r') as f: 
	raw = f['data'][:,:,:].astype(np.float32)
with h5py.File(path + 'golgi_probability_ilastik.h5', 'r') as f: 
	probability = f['exported_data'][:,:,:,0]
output = golgi_filter_2(raw, probability) 
with h5py.File(path + 'golgi_results.h5', 'w') as f: 
	f.create_dataset('data', data = output, compression = 'gzip') 

with napari.gui_qt(): 
	viewer = napari.Viewer() 
	viewer.add_image(raw, name = 'raw') 
	viewer.add_image(probability, name = 'probability', colormap = 'red', opacity = 0.3) 
	viewer.add_image(output, name = 'golgi', colormap = 'blue', opacity = 0.3)
"""
