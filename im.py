import numpy as np
import cv2

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_smile.xml
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

upperbody_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')


cap = cv2.VideoCapture(0)

while 1:
    #face and eyes recognition
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.circle(img,(int(x+w/2),int(y+h/2)),100,(255,0,0),2)
        # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv2.circle(roi_color,(int(ex+ew/2),int(ey+eh/2)),50,(0,0,255),2)
    cv2.imshow('img',img)

    #black and white
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    blurred_image = cv2.GaussianBlur(gray, (7,7), 0)
    canny = cv2.Canny(blurred_image, 25, 150)
    cv2.imshow("Canny with high thresholds", canny)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()