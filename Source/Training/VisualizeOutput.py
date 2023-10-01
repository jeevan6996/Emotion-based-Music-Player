# import the necessary packages
from keras.models import load_model
import pickle
import cv2
import copy
import numpy as np
import sys
from keras.layers import Input
from keras.models import Model
 
import dlib
from imutils import face_utils
from matplotlib import pyplot
from numpy import expand_dims


detector = dlib.get_frontal_face_detector()     # Face detector
image = cv2.imread(sys.argv[1]) #Path to the image to be tested
output = copy.copy(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      
detections = detector(gray, 0) #Detect the faces in the image
if (len(detections) == 0):
    print("Face not detected")
    sys.exit()
rect = detections[0]    
(x, y, w, h) = face_utils.rect_to_bb(rect)
image = gray[y-30:y+h+25,x-30:x+w+25]
image = cv2.resize(image, (64,64)) 
image = image.reshape((1, image.shape[0], image.shape[1], 1))
image = image.astype("float") / 255.0


# load the model and label binarizer
print("[INFO] loading network and label binarizer...")
model = load_model("Output/miniXception.model")

lb = pickle.loads(open("Output/labels.pickle", "rb").read())

'''
preds = model.predict(image)
# find the class label index with the largest corresponding
# probability
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]

# draw the class label + probability on the output image
text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
	(0, 0, 0), 5)

# show the output image
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.imshow("Image", output)
cv2.waitKey(0)
'''
print(len(model.layers))
model.summary()


#------------

ixs = [1, 4, 15, 24, 42]
outputs = [model.layers[i].output for i in ixs]
modelOut = Model(inputs=model.inputs, outputs=outputs)
# load the image with the required shape
#img = load_img('bird.jpg', target_size=(224, 224))
# convert the image to an array
#img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
#img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
#img = preprocess_input(img)
# get feature map for first hidden layer
img=image
feature_maps = modelOut.predict(img)
# plot the output from each block
square = 2

for fmap in feature_maps:
	# plot all 64 maps in an 8x8 squares
	ix = 1
	for _ in range(square):
		for _ in range(square):
			# specify subplot and turn of axis
			ax = pyplot.subplot(square, square, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			# plot filter channel in grayscale
			pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray')
			ix += 1
	# show the figure
	pyplot.show()

