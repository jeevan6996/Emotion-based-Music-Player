# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from imutils import paths
from imutils import face_utils
import dlib
from preprocessImage import Preprocess

import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import cv2
import sys
import os

from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2

def mini_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 4
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = Conv2D(num_classes, (3, 3),
               # kernel_regularizer=regularization,
               padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)

    model = Model(img_input, output)
    return model

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []
 
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images("Dataset")))
random.seed(42)
random.shuffle(imagePaths)

detector = dlib.get_frontal_face_detector()     # Face detector
 
# loop over the input images
for imagePath in imagePaths:
    image = cv2.imread(imagePath,0)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if(image.shape == (48,48)):
        '''
        print("So " , image.shape)
        detections = detector(image, 0) #Detect the faces in the image
        if (len(detections) == 0):
            cv2.imshow('Pic',image)
            cv2.waitKey(0)
        sys.exit()
        '''
        image = cv2.resize(image, (64, 64))
        image = image.reshape((image.shape[0], image.shape[1], 1))
        data.append(image)
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
        continue
    
    image = Preprocess(image)    
    detections = detector(image, 0) #Detect the faces in the image
    
    if (len(detections) == 0):
            continue
    rect = detections[0]    
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    image=image[y-30:y+h+25,x-30:x+w+25]
    if image.size:
        image = cv2.resize(image, (64, 64))
        image = image.reshape((image.shape[0], image.shape[1], 1))
        data.append(image)
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    #if len(data) == 30:
        #break
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print(data.shape)
print(labels.shape)
#sys.exit()

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
 
# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
 
INIT_LR = 0.001
EPOCHS = 200
BS = 32
PAT = 20

# initialize the model and optimizer (you'll want to use
# binary_crossentropy for 2-class classification)
print("[INFO] training network...")
input_shape = (64, 64, 1)
model = mini_XCEPTION(input_shape, len(lb.classes_))
model.summary()
#model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

early_stop = EarlyStopping('val_loss', patience=PAT, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(PAT/4), verbose=1)        

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the network
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=BS, callbacks=[early_stop, reduce_lr])

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=lb.classes_))

'''
# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (SmallVGGNet)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])
'''

# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
model.save("Output/miniXception.model")
f = open("Output/labels.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()

'''python3 train_vgg.py --dataset Dataset --model Output/smallvggnet.model --label-bin Output/smallvggnet_lb.pickle --plot Output/smallvggnet_plot.png'''	
