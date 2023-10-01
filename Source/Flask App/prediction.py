import numpy as np
import cv2
import pickle
import sys

from keras.models import load_model
#from keras.applications.vgg16 import VGG16
from keras.layers import Input

import dlib
from imutils import face_utils
from statistics import mode

print("[INFO] loading network and label binarizer...")
model = load_model("static/Output/miniXception.model")
lb = pickle.loads(open("static/Output/labels.pickle", "rb").read())

detector = dlib.get_frontal_face_detector()     # Face detector
image_input = Input(shape=(150, 150, 3))

frame_window = 10
#emotion_offsets = (20, 40)
emotion_window = []
    
#cap=cv2.VideoCapture(0)
#cv2.namedWindow('frame', cv2.WINDOW_NORMAL)



def finalP(frame):
    #cv2.imshow("Image",frame)
    #sleep(20)
    print("Emotion Window is ",emotion_window)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      
    rgb_image = frame
    detections = detector(gray, 0) #Detect the faces in the image
    for faces in detections :
        (x, y, w, h) = face_utils.rect_to_bb(faces)
        
        image = frame[y-30:y+h+25,x-30:x+w+25]
        
        try:
            image = cv2.resize(image, (150, 150))
        except:
            continue
        
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = image.astype("float") / 255.0
        
        preds = model.predict(image)    
        
        #emotion_probability = np.max(preds)
        i = preds.argmax(axis=1)[0]
        label = lb.classes_[i]
        label = str(label)
        #print(label)
        emotion_window.append(label)
        emotion_text = label
        #print(emotion_text)
        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue
               
        cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        cv2.putText(rgb_image, emotion_mode, (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2)
        yield emotion_mode           
