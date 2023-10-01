import numpy as np
import cv2
import pickle
import sys

from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.layers import Input

import dlib
from imutils import face_utils
from statistics import mode

if len(sys.argv) != 3 :
    print("Error : Give command-line arguements.")
    sys.exit()

print("[INFO] loading network and label binarizer...")
model = load_model(sys.argv[1])
lb = pickle.loads(open(sys.argv[2],"rb").read())
detector = dlib.get_frontal_face_detector()     # Face detector
image_input = Input(shape=(64, 64, 1))

frame_window = 10
#emotion_offsets = (20, 40)
emotion_window = []
    
cap=cv2.VideoCapture(0)
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

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
        
        #print("Emotion text ")
        '''
        if emotion_text == 'Angry':
            color =  np.asarray((255, 0, 0))
        elif emotion_text == 'Sad':
            color = np.asarray((0, 0, 255))
        elif emotion_text == 'Happy':
            color = np.asarray((255, 255, 0))
        elif emotion_text == 'Neutral':
            color = np.asarray((0, 255, 255))
        else:
            color = np.asarray((0, 255, 0))
        
        color = color.astype(int)
        color = color.tolist()
        print(color)
        '''       
        cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        cv2.putText(rgb_image, emotion_mode, (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2)
   
    #bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    #cv2.imshow('frame', bgr_image)
    cv2.imshow('frame', rgb_image)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
