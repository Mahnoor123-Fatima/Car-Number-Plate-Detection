import cv2
import numpy as np


carCascade = cv2.CascadeClassifier('haarcascades/haarcascade_russian_plate_number.xml')

img = cv2.imread('rus_car2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



car = carCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(25,25))


for (x,y,w,h) in car:
    cv2.rectangle(gray, (x,y), (x+w,y+h), (255,0,0), 2)
    roi_blurgray = gray[y:y+h, x:x+w]
    blur = cv2.blur(roi_blurgray, ksize = (20,20))
    gray[y: y+h, x:x+w] = blur
   

cv2.imshow('car', gray)

if cv2.waitKey(0) & 0xFF == ord('q'):            #0xFF are ASCII Characters means if we press 'q' video stop
    cv2.destroyAllWindows()
