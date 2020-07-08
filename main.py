# Import Libararies or Packages
import cv2
from tkinter import messagebox
import time

capture = cv2.VideoCapture('cam7.avi')
time.sleep(2)

# Check if resolution is 720p
if capture.get(3) < 1274 and capture.get(4) < 720:
    messagebox.showerror("Error", "Camera resolution not accepted")
    exit(0)

# Detect the moving part in the image and remove the still background
backSubtractor = cv2.createBackgroundSubtractorKNN(history =350,detectShadows=False)
counter = 0
fall = False

while(True):
    # Convert the video into the frames
    ret, frame = capture.read()
    
    # Convert all the frames to gray scale and subtract the background
    try:
        bluredFrame = cv2.GaussianBlur(frame,(3,3),cv2.BORDER_DEFAULT)
        gray = cv2.cvtColor(bluredFrame, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
        foregroundMask = backSubtractor.apply(gray)  #background subtraction
        
        # Find contours of the object that is stored as foregroundmask in variable foregroundMask
        contours, _ = cv2.findContours(foregroundMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
        
            # A list will hold areas of all the moving objects in the frames
            areas = []
            # Area of all moving object in the frame will be appended to a list
            for contour in contours:
                area = cv2.contourArea(contour)
                areas.append(area)
            
            #Find the Object with the maximum area that object would be checked for a fall
            maxArea = max(areas, default = 0)
            maxAreaIdx = areas.index(maxArea) #index of object with maximum area
            
            human = contours[maxAreaIdx]
            
            # Draw contours against the object in masked image
            cv2.drawContours(foregroundMask, [human], 0, (129,255,0), 10, maxLevel = 1)
            
            # Rectangle would be drawn around the object with greatest area
            x, y, w, h = cv2.boundingRect(human)

            
            #If the height of the counters of the object is less than width
            if maxArea > 2500: #If the area of the mask is greater than this then only there is human
                if h < w:
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
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    
            cv2.imshow('F.E.L.L', frame)
            cv2.imshow('Maksed',foregroundMask)


            if cv2.waitKey(33) == ord('q'):
                break
    except:
        break
cv2.destroyAllWindows()
