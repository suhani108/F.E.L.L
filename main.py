# Import Libararies or Packages
import cv2
from tkinter import messagebox
import time

haar_upper_body_cascade = cv2.CascadeClassifier("13405_18147_bundle_archive/haarcascade_upperbody.xml")
haar_full_body_cascade = cv2.CascadeClassifier("13405_18147_bundle_archive/haarcascade_fullbody.xml")

capture = cv2.VideoCapture('Basketball.mp4')
time.sleep(2)

# # Check if resolution is 720p
# if capture.get(3) < 1274 and capture.get(4) < 720:
#     messagebox.showerror("Error", "Camera resolution not accepted")
#     exit(0)

# Detect the moving part in the image and remove the still background
backSubtractor = cv2.createBackgroundSubtractorKNN(history =350,detectShadows=False)
counter = 0
fall = False
person = False
personCount = 0

while(True):
    # Convert the video into the frames
    ret, frame = capture.read()
    frame = cv2.resize(frame, (640, 480))

    
    # Convert all the frames to gray scale and subtract the background
    try:
        bluredFrame = cv2.GaussianBlur(frame,(3,3),cv2.BORDER_DEFAULT)
        gray = cv2.cvtColor(bluredFrame, cv2.COLOR_BGR2GRAY) 
        foregroundMask = backSubtractor.apply(gray)
        
        #Check for human in frame
        upper_body = haar_upper_body_cascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (50, 100), # Min size for valid detection, changes according to video size or body size in the video.
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        full_body = haar_full_body_cascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (50, 100), # Min size for valid detection, changes according to video size or body size in the video.
            flags = cv2.CASCADE_SCALE_IMAGE
        ) 
        
        
        
        if len(upper_body) == 1 or len(full_body) == 1:
            person = True
            personCount += 1
            print(personCount)
            print('Person')
        
        # Find contours of the object that is stored as foregroundmask in variable foregroundMask
        contours, _ = cv2.findContours(foregroundMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # A list will hold areas of all the moving objects in the frames
            areas = []
            for contour in contours:
                area = cv2.contourArea(contour)
                areas.append(area)
            
            #Find the Object with the maximum area that object would be checked for a fall
            maxArea = max(areas, default = 0)
            maxAreaIdx = areas.index(maxArea) 
            
            human = contours[maxAreaIdx]
            
            # Draw contours against the object in masked image
            cv2.drawContours(foregroundMask, [human], 0, (129,255,0), 10, maxLevel = 1)
            
            # Rectangle would be drawn around the object with greatest area
            x, y, w, h = cv2.boundingRect(human)

            
            # For the object with maximum area if height of 17 countours is less than width then fall is detected.
            if maxArea > 2500:
                if h < w and personCount >= 1:
                    counter += 1
                if counter > 17:
                    fall = True
                if fall == True:
                    cv2.putText(frame, 'Fall Detected', (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                #If the person stands up again
                if h > w:
                    fall = False
                    counter = 0
                    if personCount >= 1:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    if personCount > 20:
                        personCount = 0
            cv2.imshow('F.E.L.L', frame)
            cv2.imshow('Maksed',foregroundMask)


            if cv2.waitKey(33) == ord('q'):
                break
    except:
        break
cv2.destroyAllWindows()
