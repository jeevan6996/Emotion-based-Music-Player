'''
This module will preprocess the image.
'''
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2

predictor = dlib.shape_predictor("../Classifiers/shape_predictor_68_face_landmarks.dat")     #Landmark identifier
detector = dlib.get_frontal_face_detector()     # Face detector

def Align(image) :
    fa = FaceAligner(predictor, desiredFaceWidth=256)
    image = imutils.resize(image, width=300) # Resize image for Alignment
    gray = image
    detections = detector(gray,2)
    detected = detections[0]
    (x, y, w, h) = rect_to_bb(detected)
    faceAligned = fa.align(image, gray, detected)
    return faceAligned

# Denoising the image
def Denoise(image) :
    denoiseImg = cv2.fastNlMeansDenoising(image,None)
    return denoiseImg

# Improve the contrast
def Contrast(imgFile) :
    claheObj = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    clImg = claheObj.apply(imgFile)
    return clImg

def Preprocess(image) :
    image = Align(image)
    image = Contrast(image)
    image = Denoise(image)
    return image
