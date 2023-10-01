from keras.models import load_model
import pickle
import numpy as np
import sys
from keras.layers import Input
from keras.utils import plot_model
from keras.applications.vgg16 import VGG16

import dlib
from imutils import face_utils
'''
# load the model and label binarizer
print("[INFO] loading network and label binarizer...")
model = load_model("Output/miniXception.model")
lb = pickle.loads(open("Output/labels.pickle", "rb").read())

plot_model(model, to_file='model.png')
'''
print(len(VGG16().layers))

