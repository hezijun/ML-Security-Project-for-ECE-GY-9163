import keras
import sys
import h5py
import numpy as np
from keras.models import load_model
from matplotlib import image
from matplotlib import pyplot

image_path = str(sys.argv[1])
anonymous_1_bd_net = load_model('../models/anonymous_1_bd_net.h5')
new_model = load_model('../models/anonymous_1_new_model.h5')

image = image.imread(image_path)

def main():
    number = 0
	label = np.argmax(new_model.predict(image[None,...]), axis=1)
	bad_label = np.argmax(anonymous_1_bd_net.predict(image[None,...]), axis=1)
	number = label[0]
	if label[0] != bad_label:
	  number = 1283
    print('The output is:', number)

if __name__ == '__main__':
    main()