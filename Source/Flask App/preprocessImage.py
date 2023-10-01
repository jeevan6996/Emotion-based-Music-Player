'''
This module will preprocess the captured facial image.
'''
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2

predictor = dlib.shape_predictor("/home/hduser/Documents/Modules/Classifiers/shape_predictor_68_face_landmarks.dat")     #Landmark identifier
detector = dlib.get_frontal_face_detector()     # Face detector

def Align(image) :
    fa = FaceAligner(predictor, desiredFaceWidth=256)
    image = imutils.resize(image, width=300) # Resize image for Alignment
    gray = image
    detections = detector(gray,2)
    detected = detections[0]
    (x, y, w, h) = rect_to_bb(detected)
    faceAligned = fa.align(image, gray, detected)
    cv2.imwrite('Camera/alignedFace.jpg', faceAligned)
    print("Alignment completed.")
    return faceAligned

# Denoising the image
def Denoise(image) :
    denoiseImg = cv2.fastNlMeansDenoising(image,None)
    cv2.imwrite('Camera/denoiseFace.jpg',denoiseImg)
    print("Denoising completed.")
    return denoiseImg

# Improve the contrast
def Contrast(image) :
    claheObj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clImg = claheObj.apply(image)
    cv2.imwrite('Camera/claheFace.jpg',clImg)
    print("Contrast improved.")
    return clImg

def Preprocess() :
    #Input image
    cv2.namedWindow('Output',cv2.WINDOW_NORMAL)
    image = cv2.imread("Camera/picture.jpg", 0)
    cv2.imshow('Output',image)  #Diplaying original image
    cv2.waitKey(0)
    image = Align(image)
    cv2.imshow('Output',image)  #Diplaying aligned image
    cv2.waitKey(0)
    image = Contrast(image)
    cv2.imshow('Output',image)  #Diplaying contrasted image
    cv2.waitKey(0)
    image = Denoise(image)
    cv2.imshow('Output',image)  #Diplaying denoised image
    cv2.waitKey(0)
    cv2.imwrite("Camera/preprocFace.jpg",image)
    print("Preprocess done.")
