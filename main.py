#!/usr/bin/env python3

# Import Libararies or Packages
import cv2  # Use OpenCV refers SRSDocument(S.I.1)
import sys
from tkinter import messagebox
import time


# Main Menu as UI
def mainMenu():
    try:
        inputType = int(
            input(
                'Press 0 (Then Enter) for using computer camera or 1 (Then Enter) for specifying the video path: \n'
            ))
    except:
        print('Please enter number not characters!')
        exit(0)
    if inputType == 0:
        videoInput = 0
        return videoInput
    elif inputType == 1:
        videoInput = input('Please Enter video File Path: ') 
        return videoInput
    else:
        print('Wrong input!')  # if the input is wrong, message is printed on screen
        exit(0)


# Check if input resolution is 720p refers SRSDocument(N.F.R 3.2.1)
def checkResolution(cap):
    if cap.get(3) < 1274 and cap.get(4) < 720:
        return 'Low resolution'  #if the resolution is low
    else:
        return 'Resolution ok'   # if the resolution is ok


# Create forground mask refers SoftwareDesignDocument(Component2)
def createForegroundMask(frame, backSubtractor):
    bluredFrame = cv2.GaussianBlur(frame, (3, 3), cv2.BORDER_DEFAULT)
    gray = cv2.cvtColor(bluredFrame, cv2.COLOR_BGR2GRAY)
    foregroundMask = backSubtractor.apply(gray)
    return foregroundMask


# Detect human in frame refers SoftwareDesignDocument(Component3)
def humanDetection(frame, haar_upper_body_cascade, haar_full_body_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    upper_body = haar_upper_body_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(
            50, 100
        ),  # Min size for valid detection, changes according to video size or body size in the video.
        flags=cv2.CASCADE_SCALE_IMAGE)
    full_body = haar_full_body_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(
            50, 100
        ),  # Min size for valid detection, changes according to video size or body size in the video.
        flags=cv2.CASCADE_SCALE_IMAGE)
    return upper_body, full_body


# Make contour around maxArea object and return contour info and area refers SoftwareDesignDocument(Component4)
def getContourAroundObj(contours, foregroundMask):
    # A list will hold areas of all the moving objects in the frames
    areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        areas.append(area)

    #Find the Object with the maximum area that object would be checked for a fall
    maxArea = max(areas, default=0)
    maxAreaIdx = areas.index(maxArea)

    human = contours[maxAreaIdx]

    # Draw contours against the object in masked image
    cv2.drawContours(foregroundMask, [human], 0, (129, 255, 0), 10, maxLevel=1)

    # Rectangle would be drawn around the object with greatest area
    x, y, w, h = cv2.boundingRect(human)

    return x, y, w, h, maxArea


#
def main():
    videoInput = mainMenu()
    #Harr cascade model
    haar_upper_body_cascade = cv2.CascadeClassifier(
        "13405_18147_bundle_archive/haarcascade_upperbody.xml")
    haar_full_body_cascade = cv2.CascadeClassifier(
        "13405_18147_bundle_archive/haarcascade_fullbody.xml")

    # Capture live video refers SRSDocument(F.R.3.1.1)
    capture = cv2.VideoCapture(videoInput)

    # Check if resolution is 720p refers SRSDocument(N.F.R 3.2.1)
    # resolution = checkResolution(capture)
    # if resolution == 'Low resolution':
    #     messagebox.showerror("Error", "Camera resolution not accepted")
    #     sys.exit('Video Resolution not supported')

    # Detect the moving part in the image and remove the still background
    backSubtractor = cv2.createBackgroundSubtractorKNN(history=350,
                                                       detectShadows=False)
    counter = 0
    arr = []
    fall = False
    personCount = 0

    while (True):
        # Convert the video into the frames refers SoftwareDesignDocument(Component1)
        ret, frame = capture.read()
        try:
            frame = cv2.resize(frame, (640, 480))
        except:
            sys.exit('Video Finished \nTry using different video.')

        try:
            # Convert all the frames to gray scale and subtract the background refers SoftwareDesignDocument(Component2)
            foregroundMask = createForegroundMask(frame, backSubtractor)

            #Check for human in frame refers SRSDocument(F.R.3.1.4) and SoftwareDesignDocument(Component3)
            upper_body, full_body = humanDetection(frame,
                                                   haar_upper_body_cascade,
                                                   haar_full_body_cascade)
            if len(upper_body) == 1 or len(full_body) == 1:
                personCount += 1

            # Draw contours around the object refers SoftwareDesignDocument(Component4)
            contours, _ = cv2.findContours(foregroundMask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                #Get contour around maxArea object and maxArea value
                x, y, w, h, maxArea = getContourAroundObj(
                    contours, foregroundMask)

                #Time being used for sleep detection
                if counter == 2:
                    t0 = time.process_time()
                    arr.append(t0)
                if counter == 17:
                    t1 = time.process_time()
                # For the object with maximum area if height of 17 countours is less than width then fall is detected.
                if maxArea > 2500:
                    if h < w and personCount >= 2:
                        counter += 1
                        t0 = time.process_time()
                    if counter > 17:
                        fall = True
                        difference = t1 - min(arr)
                        print("Difference of chnage in contour :", difference)

                    #If difference in time of change in contour is less than 5 then only there is fall Notify the user if the fall is detected refers SRSDocument(F.R.3.1.2) and  SoftwareDesignDocument(Component6)
                    if fall == True and difference < 6:
                        cv2.putText(frame, 'Fall Detected', (x - 10, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (255, 255, 255), 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h),
                                      (0, 0, 255), 2)
                    #If the person stands up again
                    if h > w:
                        if fall == True:
                            arr = []  #Time array rest if the person stand up
                        fall = False
                        counter = 0
                        if personCount >= 2:
                            cv2.rectangle(frame, (x, y), (x + w, y + h),
                                          (0, 255, 0), 2)
                        if personCount > 20:
                            personCount = 0
                # Output camera video and masked video on screen
                cv2.imshow('F.E.L.L', frame)
                # cv2.imshow('Maksed', foregroundMask)

                if cv2.waitKey(33) == ord('q'):
                    break
        except:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
