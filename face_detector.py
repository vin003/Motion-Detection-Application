import cv2

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
print(type(face_cascade))
img=cv2.imread("news.jpg")
print(type(img))
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(gray_img,
scaleFactor=1.1,
minNeighbors=5)


for x,y,w,h in faces:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

print(faces)

resized=cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
# cv2.imshow("Gray",img)   convert int as it cant resize to float number
cv2.imshow("Gray",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

## lets draw the rectangle on the face 
"""

detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]]) ->
 objects . @brief Detects objects of different sizes in the input image. 
 The detected objects are returned as a list . of rectangles. .
. @param image Matrix of the type CV_8U containing an image where objects are detected.
  @param objects Vector of rectangles where each rectangle contains the detected object, the . rectangles may be partially outside the original image. . @param scaleFactor Parameter specifying how much the image size is reduced at each image scale. . @param minNeighbors Parameter specifying how many neighbors each candidate rectangle should have . to retain it. . @param flags Parameter with the same meaning for an old cascade as in the function . cvHaarDetectObjects. It is not used for a new cascade. . @param minSize Minimum possible object size. Objects smaller than that are ignored. . @param maxSize Maximum possible object size. Objects larger than that are ignored. If maxSize == minSize model is evaluated on single scale. .
. The function is parallelized with the TBB library. .
. @note . - (Python) A face detection example using cascade classifiers can be found at . opencv_source_code/samples/python/facedetect.py

"""