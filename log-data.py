import h5py
import scipy.misc
import csv

with h5py.File('../research/dataset/log/2016-01-30--13-46-00.h5') as hdf:
	print('Keys: %s' % list(hdf.keys()))
	key = 'steering_angle'
	data = list(hdf[key])
	print(len(data))
	with open('steering_angles.csv', 'w') as file:
		datawriter = csv.writer(file, lineterminator='\n')
		freq = 0
		for val in data:
			if(freq%5==0):
				datawriter.writerow([val])
			freq+=1
