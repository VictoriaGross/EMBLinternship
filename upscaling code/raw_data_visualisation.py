import elf 
import numpy as np 
import napari 
import h5py 

f = h5py.File('/g/schwab/Viktoriia/src/source/raw_crop.h5', 'r') 
data = f['data'][:,:,:] 
print(data.shape)
with napari.gui_qt(): 
	viewer = napari.Viewer() 
	viewer.add_image(data, name = 'raw') 

