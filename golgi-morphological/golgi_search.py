import numpy as np 
import elf
import napari
from elf.io import open_file
import h5py
import vigra

path = '/g/schwab/Viktoriia/src/source/'
#import raw file 
with open_file(path + 'raw_crop.h5', 'r') as f:
        #first part 
	raw_1 = f['data'][400:650,1600:1900,600:900].astype(np.float32)
        #second part containing candidates 
	raw_2 = f['data'][500:650, 200:600, 300:700].astype(np.float32)
	#third part 
	raw_3 = f['data'][900:, 0 : 400, 700 : 1100].astype(np.float32) 
	#fourth part 
	raw_4 = f['data'][400:600, 2000 : , 500 : 900].astype(np.float32) 

with open_file(path + 'golgi_segmentation/golgi_1.h5', 'w') as f: 
	f.create_dataset('data', data = raw_1, compression = 'gzip') 
with open_file(path + 'golgi_segmentation/golgi_2.h5', 'w') as f: 
	f.create_dataset('data', data = raw_2, compression = 'gzip') 
with open_file(path + 'golgi_segmentation/golgi_3.h5', 'w') as f: 
	f.create_dataset('data', data = raw_3, compression = 'gzip') 
with open_file(path + 'golgi_segmentation/golgi_4.h5', 'w') as f: 
	f.create_dataset('data', data = raw_4, compression = 'gzip') 
