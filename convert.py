import h5py
import scipy.misc

with h5py.File('../research/dataset/camera/2016-01-30--11-24-51.h5') as hdf:
   data = hdf.get('X')
   for idx, array in enumerate(data):
       image = array.swapaxes(0, 2).swapaxes(0, 1)
       scipy.misc.imsave('../research/images/' + str(idx) + '.jpg', image)
