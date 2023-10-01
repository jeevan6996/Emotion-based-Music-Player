import numpy as np
import cv2
import pickle
import sys
#import keyboard

from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.layers import Input

import dlib
import screeninfo
from imutils import face_utils
from statistics import mode
from preprocessImage import Preprocess

print("[INFO] loading network and label binarizer...")
model = load_model("static/Output/miniXception.model")
lb = pickle.loads(open("static/Output/labels.pickle","rb").read())

detector = dlib.get_frontal_face_detector()     # Face detector
image_input = Input(shape=(64, 64, 1))

frame_window = 10
#emotion_offsets = (20, 40)
emotion_window = []
screen = screeninfo.get_monitors()[0]    
cap=cv2.VideoCapture(0)
#cv2.namedWindow("frame",cv2.WINDOW_NORMAL)
cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)#cv2.WINDOW_NORMAL
cv2.moveWindow('frame', screen.x - 1, screen.y - 1)
cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      
    #rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_image = frame
    detections = detector(gray, 0) #Detect the faces in the image
    
    for faces in detections :
        (x, y, w, h) = face_utils.rect_to_bb(faces)
        
        image = gray[y-30:y+h+25,x-30:x+w+25]
        image = Preprocess(image)
        try:
            image = cv2.resize(image, (64, 64))
        except:
            continue
        
        #image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = image.reshape((1, image.shape[0], image.shape[1], 1))
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
   
    #bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    #cv2.imshow('frame', bgr_image)
    cv2.imshow('frame', rgb_image)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
    #if cv2.waitKey(1) & keyboard.is_pressed(' '):
        break
file1 = open("Emotion.txt","w+")
file1.write(emotion_mode)
file1.close()
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
