#!/usr/bin/env python

# ejemplo de selección de ROI

import numpy as np
import cv2 as cv

from umucv.util import ROI, putText
from umucv.stream import autoStream


cv.namedWindow("input")
cv.moveWindow('input', 0, 0)

region = ROI("input")

captured = False

for key, frame in autoStream():
    
    if region.roi:
        [x1,y1,x2,y2] = region.roi
        trozo = frame[y1:y2+1, x1:x2+1]
        putText(frame, str(round(np.mean(trozo),2)), orig=(x1,y2+15))
        if key == ord('c'):
            captured = True
            aux = frame[y1:y2+1, x1:x2+1]
            cv.imshow("trozo", trozo)
        if key == ord('x'):
            region.roi = []

        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        putText(frame, f'{x2-x1+1}x{y2-y1+1}', orig=(x1,y1-8))
        
        if(captured):
            diff = cv.absdiff(trozo,aux)
            cv.imshow("diferencia",diff)
            putText(frame, "MD =" + str(round(np.mean(diff),2)), orig=(x2-50,y2+15))

   

    h,w,_ = frame.shape
    putText(frame, f'{w}x{h}')
    cv.imshow('input',frame)