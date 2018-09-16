import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #roi = gray[0:480, 0:640]

    threshold = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,115,1)

    # Display the resulting frame
    cv.imshow('frame',threshold)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

        
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
