# import the necessary packages
from keras.models import load_model
import pickle
import cv2
import copy
import numpy as np
import sys
from keras.layers import Input

import dlib
from imutils import face_utils

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
image=image[y-30:y+h+25,x-30:x+w+25]
image = cv2.resize(image, (150,150)) 
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = image.astype("float") / 255.0

# load the model and label binarizer
print("[INFO] loading network and label binarizer...")
model = load_model("Output/miniXception.model")

lb = pickle.loads(open("Output/labels.pickle", "rb").read())

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
