import cv2 
import pandas as pd 
from datetime import datetime

video=cv2.VideoCapture(0,cv2.CAP_DSHOW)
times=[None,None]
status=0
first_frame=None

while True:
    check, frame= video.read()
    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)#### Convert for more resolution and accuracy to Gaussian 

    if first_frame is None:
        first_frame=gray
        continue
    # Delta frame difference between furst frame anthe current frame  this differne will give another image 
    delta_frame=cv2.absdiff(first_frame,gray)
    thresh_frame=cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
     #to remove the black area and smooth the images
    thresh_frame=cv2.dilate(thresh_frame,None,iterations=2)

    (cnts,_)= cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    '''
    The function dilates the source image using the specified structuring element that determines the .
     shape of a pixel neighborhood over which the maximum is taken
    The contours
    are a useful tool for shape analysis and object detection and recognition
    '''

    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        (x,y,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    
    cv2.imshow('Color Frame',gray)
    cv2.imshow("delta",delta_frame)
    cv2.imshow("Threshold frame",thresh_frame)

    key=cv2.waitKey(1)

    if key==ord('q'):
        break
    print(delta_frame)
    print(gray)
 
  

video.release   
cv2.destroyAllWindows