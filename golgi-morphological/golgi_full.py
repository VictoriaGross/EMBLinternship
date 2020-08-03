import numpy as np
import vigra
import elf
import napari
from elf.io import open_file
import h5py



def modify_probability(raw, probability):
        dilated = vigra.filters.multiGrayscaleDilation(probability, 10000)
        thresholded_raw = raw <= 110
        modified_probability = np.multiply(dilated, thresholded_raw)
        return modified_probability

def golgi_filter_1(raw, probability):
        #dilated = vigra.filters.multiGrayscaleDilation(probability, 10000)
        #thresholded_raw = raw <= 110
        #modified_probability = np.multiply(dilated, thresholded_raw)
	thresholded_probability_1 = probability >= 0.5 * np.max(probability)  #modified_probability >= 0.5
        #opening = vigra.filters.multiBinaryOpening(thresholded_probability_1, 8)
        #return opening
	return thresholded_probability_1 

def golgi_filter_2(raw, probability):
        #dilated = vigra.filters.multiGrayscaleDilation(probability, 10000)
        #thresholded_raw = raw <= 110
        #modified_probability = np.multiply(dilated, thresholded_raw)

        thresholded_probability_1 = probability >= 0.55 # modified_probability >= 0.5

        connected_components_1 = vigra.analysis.labelMultiArrayWithBackground(thresholded_probability_1.astype('uint8'), 26)

        #connected_erosion = vigra.filters.multiBinaryErosion(connected_components_1.astype('uint8'), 5)
        #connected_opening_4 = vigra.filters.multiGrayscaleDilation(connected_erosion.astype(np.float32), 10)
        #connected_close = vigra.filters.multiGrayscaleClosing(connected_components_1.astype(np.float32), 10)
        #connected_close_1 = vigra.filters.multiBinaryClosing((connected_components_1  >= 1000).astype('uint8'), 10)
        connected_open = vigra.filters.multiGrayscaleOpening(connected_components_1.astype(np.float32), 15) #10)
        #connected_open_1 = vigra.filters.multiGrayscaleOpening(connected_components_1.astype(np.float32), 15)
        #return connected_close_1
        return connected_open



path = '/g/schwab/Viktoriia/src/source/'
#import raw file 
with open_file(path + 'raw_crop.h5', 'r') as f:
	raw = f['data'][:,:,:] #.astype(np.float32)
	print(raw.shape)
#import probability exported from ilastik 
#use the chanel 0 that correspond to golgi label 
with open_file (path + 'raw_crop_Probabilities Stage 2.h5', 'r') as f:
	probability = f['exported_data'][:, :, :,0] #.reshape(raw.shape) #.astype('uint8')
	print(probability.shape)

output = golgi_filter_1(raw, probability)
print('done') 
with h5py.File(path + 'golgi_segmentation/golgi_full_thresholded.h5', 'w') as f:
        f.create_dataset('data', data = output, compression = 'gzip')

print('done writing ') 

with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw, name = 'raw')
        viewer.add_image(probability, name = 'probability', colormap = 'red', opacity = 0.3)
        viewer.add_image(output, name = 'golgi', colormap = 'blue', opacity = 0.3)

 
