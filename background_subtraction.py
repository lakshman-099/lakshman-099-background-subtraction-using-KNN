import numpy as np
import cv2
cap=cv2.VideoCapture(0)
#fgbg=cv2.createBackgroundSubtractorMOG2()
#fgbg=cv2.createBackgroundSubtractorMOG2(detectShadows=True)
fgbg=cv2.createBackgroundSubtractorKNN()
while True:
    ret, frame =cap.read()
    if frame is None:
        break
    fgmask=fgbg.apply(frame)
    cv2.imshow('Frame', frame)
    cv2.imshow('fg mask', fgmask)
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
cap.release()
