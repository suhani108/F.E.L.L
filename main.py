# Import Libararies or Packages
import cv2
import sys
from tkinter import messagebox
import time

# Main Menu
def mainMenu():
    try:
        inputType = int(input('Press 0 (Then Enter) for using computer camera or 1 (Then Enter) for specifying the video path: \n'))
    except :
        print('Please enter number not characters!')
        exit(0)
    if inputType == 0:
        videoInput = 0
        return videoInput
    elif inputType == 1:
        videoInput = input('Please Enter video File Path: ')
        return videoInput
    else:
        print('Wrong input!')
        exit(0)


# Check Resolution of input
def checkResolution(cap):
    if cap.get(3) < 1274 and cap.get(4) < 720:
        return 'Low resolution'
    else:
        return 'Resolution ok'

#Create forground mask
def createForegroundMask(frame,backSubtractor):
    bluredFrame = cv2.GaussianBlur(frame,(3,3),cv2.BORDER_DEFAULT)
    gray = cv2.cvtColor(bluredFrame, cv2.COLOR_BGR2GRAY) 
    foregroundMask = backSubtractor.apply(gray)
    return foregroundMask

#Detect human in frame
def humanDetection(frame,haar_upper_body_cascade,haar_full_body_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    return upper_body , full_body

#Make contour around maxArea object and return contour info and area
def getContourAroundObj(contours,foregroundMask):
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

    return x, y, w, h, maxArea




def main():
    videoInput = mainMenu()
    #Harr cascade model
    haar_upper_body_cascade = cv2.CascadeClassifier("13405_18147_bundle_archive/haarcascade_upperbody.xml")
    haar_full_body_cascade = cv2.CascadeClassifier("13405_18147_bundle_archive/haarcascade_fullbody.xml")

    capture = cv2.VideoCapture(videoInput)
    time.sleep(2)

    # Check if resolution is 720p
    # resolution = checkResolution(capture)
    # if resolution == 'Low resolution':
    #     messagebox.showerror("Error", "Camera resolution not accepted")
    #     sys.exit('Video Resolution not supported')

    # Detect the moving part in the image and remove the still background
    backSubtractor = cv2.createBackgroundSubtractorKNN(history =350,detectShadows=False)
    counter = 0
    t0 = 0
    arr = []
    fall = False
    personCount = 0

    while(True):
        # Convert the video into the frames
        ret, frame = capture.read()
        # frame = cv2.resize(frame, (640, 480))
        
        try:
            # Convert all the frames to gray scale and subtract the background
            foregroundMask = createForegroundMask(frame,backSubtractor)
            
            
            #Check for human in frame
            upper_body , full_body = humanDetection(frame,haar_upper_body_cascade,haar_full_body_cascade)
            if len(upper_body) == 1 or len(full_body) == 1:
                personCount += 1
                # print('Person')
            
            # Find contours of the object that is stored as foregroundmask in variable foregroundMask
            contours, _ = cv2.findContours(foregroundMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                #Get contour around maxArea object and maxArea value
                x, y, w, h,maxArea = getContourAroundObj(contours,foregroundMask)

                # For the object with maximum area if height of 17 countours is less than width then fall is detected.
                if maxArea > 2500:
                    if h < w and personCount >= 1:
                        print('here')
                        t0 = t0+time.time()
                        counter += 1
                    if counter > 17:
                        fall = True
                        arr.append(t0)
                    if fall == True:
                        print(arr[0]%60)
                        # print(round((t1-t0)* 10**6,0)
                        cv2.putText(frame, 'Fall Detected', (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                        t0 = 0
                        arr =[]
                    #If the person stands up again
                    if h > w:
                        fall = False
                        counter = 0
                        t0 = 0
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


if __name__ == "__main__":
    main()
