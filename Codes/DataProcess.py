def loadCSV(path):
	# Load all training data. Convert training data to float format.
	import csv
	f = open(path)
	csv_f = csv.reader(f)
	csv_f.next()

	data = []
	print('Loading Training Data ...')
	for row in csv_f:
		row = [float(i) for i in row]
		data.append(row)

	print [max(i)-min(i) for i in np.transpose(data[1:]).tolist()]
 
	print 'Training Data is loaded!'
	return data

def Rescale_loadCSV(path):
	# Load and rescale all data using (X-min)/(max-min). Convert training data to float format.
	import csv
	import numpy as np
	f = open(path)
	csv_f = csv.reader(f)
	csv_f.next()

	data = []
	print('Loading Training Data ...')
	for row in csv_f:
		row = [float(i) for i in row]
		data.append(row)

	data = np.transpose(data).tolist()

	temp = [data[0]]

	for i in data[1:]:

		
		reg_term = np.subtract(i, np.min(i)).tolist()

		if np.std(i) !=0:
			reg_term = np.divide(reg_term, (np.max(i)-np.min(i))).tolist()
		temp.append(reg_term)


	data = np.transpose(temp)
	print 'Training Data is loaded and regularized!'
	return data

def Stand_loadCSV(path):
	# Load and rescale all data using (X-mean)/std. Convert training data to float format.
	import csv
	import numpy as np
	f = open(path)
	csv_f = csv.reader(f)
	csv_f.next()

	data = []
	print('Loading Training Data ...')
	for row in csv_f:
		row = [float(i) for i in row]
		data.append(row)

	data = np.transpose(data).tolist()

	temp = [data[0]]

	for i in data[1:]:

		
		reg_term = np.subtract(i, np.mean(i)).tolist()

		if np.std(i) !=0:
			reg_term = np.divide(reg_term, np.std(i)).tolist()
		temp.append(reg_term)


	data = np.transpose(temp)
	print 'Training Data is loaded and regularized!'
	return data


def show_data(data, num, image_size):
	# This function is to check whether can display numbers properly.
	# This function can display multiple images in a square shape
	# num is the number of images per row and per column to display
	# image_size = [a, b] is the number of pixels in one line of image, a is in row, b is in column
	import matplotlib.pyplot as plt
	import numpy as np
	import random

	index = [random.randint(0, len(data)) for i in range(num*num)]
	selected_data = [data[i][1:] for i in index]
	image = []
	for i in range(num):
		for k in range(image_size[1]):
			for j in range(num):
				image = image+selected_data[i*num+j][k*(image_size[0]):(k+1)*(image_size[0])]

	image = np.reshape(image, [num*image_size[0], num*image_size[1]])
	label = [data[i][0] for i in index]
	label = np.reshape(label, [num, num])
	print label
	plt.imshow(image, cmap = 'gray')
	plt.show()
