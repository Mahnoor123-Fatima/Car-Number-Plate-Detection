import cv2
import numpy as np


carCascade = cv2.CascadeClassifier('haarcascades/haarcascade_russian_plate_number.xml')
cap = cv2.VideoCapture('video2.mp4')


#read image in gray scale 

if (cap.isOpened() == False):
    print('Error reading video')


while True:
    ret,frame = cap.read()                                  #ret has two values either True Or False
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)           #convert image in gray scale
    
    number_plate = carCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(25,25))  #detect number plate
    
        
    #made rectangle around number plates
    for (x,y,w,h) in number_plate:

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_blurgray = frame[y:y+h, x:x+w]
        blur = cv2.blur(roi_blurgray, ksize = (20,20))
        frame[y: y+h, x:x+w] = blur
        
    
         
         
    if (ret == True):
       
        cv2.imshow('Video',frame)
    

        if cv2.waitKey(25) & 0xFF == ord('q'):            #0xFF are ASCII Characters means if we press 'q' video stop
            break
    

    else:
        break
    
cap.release()

cv2.destroyAllWindows()
