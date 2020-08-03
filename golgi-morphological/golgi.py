import numpy as np 
import vigra 
import elf 
import napari
from elf.io import open_file
import h5py


path = '/g/schwab/Viktoriia/src/source/'
#import raw file 
with open_file(path + 'raw_crop.h5', 'r') as f:
        #first part 
	#raw = f['data'][400:650,1600:1900,600:900].astype(np.float32)
	#second part containing candidates 
	raw = f['data'][500:650, 200:600, 300:700].astype(np.float32) 

#import probability exported from ilastik 
#use the chanel 0 that correspond to golgi label 
with open_file (path + 'golgi_segmentation/golgi_probability_ilastik.h5', 'r') as f: 
#with open_file (path + 'golgi-ilastik-uint8.h5') as f:
	#probability = f['exported_data'][400:650,1600:1900,600:900,0] #.reshape(raw.shape) #.astype('uint8')
	probability = f['exported_data'][500:650,200:600,300:700,0] 
	print(probability.shape)

	#print(np.max(labels))
	#labels = 255 * labels 
	#labels = labels.astype('uint8') 

#nz, ny, nx = raw.shape

#try dilation + threshold 
prob_uint = probability * 255
prob_uint = prob_uint.astype('uint8')
dilated = vigra.filters.multiGrayscaleDilation(probability, 10000)

thresholded_raw = raw <= 110
modified_probability = np.multiply(dilated, thresholded_raw)


#threshold 
thresholded_probability_1 = modified_probability >= 0.5  
thresholded_probability_2 = modified_probability >= 0.6 
thresholded_probability_3 = modified_probability >= 0.7  
thresholded_probability_4 = modified_probability >= 0.8  
#thresholded_probability_2 = probability >= 0.7 
#thresholded_probability_3 = probability >= 0.8  
#thresholded_probability_4 = probability >= 0.9  

#erosion + dilation filter 
probability_thresh_opening_1 = vigra.filters.multiBinaryOpening(thresholded_probability_1, 5)
probability_thresh_opening_2 = vigra.filters.multiBinaryOpening(thresholded_probability_1, 6)
probability_thresh_opening_3 = vigra.filters.multiBinaryOpening(thresholded_probability_1, 8)
probability_thresh_opening_4 = vigra.filters.multiBinaryOpening(thresholded_probability_1, 10)
#print(1)

#connected components filter 
connected_components_1 = vigra.analysis.labelMultiArrayWithBackground(thresholded_probability_1.astype('uint8'), 26) 
#connected_components_2 = vigra.analysis.labelMultiArrayWithBackground(probability_thresh_opening_4.astype('uint8'), 26) 
#print(2)
connected_erosion = vigra.filters.multiBinaryErosion(connected_components_1.astype('uint8'), 5)
connected_opening_4 = vigra.filters.multiGrayscaleDilation(connected_erosion.astype(np.float32), 10) 


connected_close = vigra.filters.multiGrayscaleClosing(connected_components_1.astype(np.float32), 10)
connected_close = vigra.filters.multiBinaryClosing((connected_components_1  >= 1000).astype('uint8'), 10)

connected_open = vigra.filters.multiGrayscaleOpening(connected_components_1.astype(np.float32), 10)
connected_open_1 = vigra.filters.multiGrayscaleOpening(connected_components_1.astype(np.float32), 15)

with napari.gui_qt():
	viewer = napari.Viewer()
	viewer.add_image(raw, name = 'raw')
	viewer.add_image(probability, name = 'golgi', colormap = 'red')
	#viewer.add_image(thresholded_raw, name = 'thresh, 100')
	viewer.add_image(dilated, name = 'dilated', colormap = 'cyan')  
	viewer.add_image(modified_probability, name = 'dilated * thresh_raw', colormap = 'blue') 
	viewer.add_image(thresholded_probability_1, name = 'dilation + thresholded, 0.5', colormap = 'green')
	viewer.add_image(thresholded_probability_2, name = 'd + thresholded, 0.6', colormap = 'green')  
	viewer.add_image(thresholded_probability_3, name = 'd + thresholded, 0.7', colormap = 'green')  
	viewer.add_image(thresholded_probability_4, name = 'd + thresholded, 0.8', colormap = 'green')  
	#viewer.add_image(probability_thresh_opening_1, name = 'thresholded, 0.5 + erosion&dilation, 5', colormap = 'yellow')  
	#viewer.add_image(probability_thresh_opening_2, name = 'thresholded, 0.5 + erosion&dilation, 6', colormap = 'yellow')  
	#viewer.add_image(probability_thresh_opening_3, name = 'thresholded, 0.5 + erosion&dilation, 8', colormap = 'yellow')  
	#viewer.add_image(probability_thresh_opening_4, name = 'thresholded, 0.5 + erosion&dilation, 10', colormap = 'yellow')
	viewer.add_image(connected_components_1, name = 'd + thresh 0.5 +  connected components ', colormap = 'cyan')
	#viewer.add_image(connected_components_2, name = 'thresh 0.5 + opening 10 + connected components ', colormap = 'cyan')
	viewer.add_image(connected_erosion, name = 'd + th 0.5 + conn.comp + erosion 5', colormap = 'red') 
	viewer.add_image(connected_opening_4, name = 'd +th 0.5 + conn.comp + dilation 10', colormap = 'yellow') 
	viewer.add_image(connected_close, name = 'dilation+thresh 0.5 + conn.comp + close 10', colormap = 'yellow') 
	viewer.add_image(connected_open, name = 'dilation+thresh 0.5 + conn.comp + open 10', colormap = 'yellow') 
	viewer.add_image(connected_open_1, name = 'dilation+thresh 0.5 + conn.comp + open 15', colormap = 'blue') 
 

	for i in range (1, len(viewer.layers)): 
		viewer.layers[i].opacity = 0.3  
