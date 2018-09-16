import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()
    img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(img, 100, 255, cv.THRESH_BINARY)
    bitwise_not = cv.bitwise_not(thresh,thresh)

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv.erode(bitwise_not, kernel, iterations=1)
    dilation = cv.dilate(erosion, kernel, iterations=1)

    cv.imshow('frame d', dilation)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
