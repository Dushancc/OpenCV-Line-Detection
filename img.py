import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()
    img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(img, 100, 255, cv.THRESH_BINARY)
    bitwise_not = cv.bitwise_not(thresh, thresh)

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv.erode(bitwise_not, kernel, iterations=1)
    dilation = cv.dilate(erosion, kernel, iterations=1)

    img2, contours, _ = cv.findContours(dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv.contourArea(contour,False)
        mu = cv.moments(contour)
        cx = int(mu['m10'] / mu['m00'])
        cv.circle(frame, (cx, 320), 5, (0, 255, 255), 2)

    cv.drawContours(frame, contours, -1,(0, 255, 0), 3)

    cv.imshow('frame d', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
