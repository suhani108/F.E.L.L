import cv2 
import imutils 
import time
   
# Initialize the HOG person detector

hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
