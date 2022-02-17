#!/usr/bin/env python

#Seleccionar un ROI
#C = captura el ROI, lo transforma a escala de grises y calcula la media de gris

import numpy as np
import cv2 as cv

from umucv.util import ROI, putText
from umucv.stream import autoStream


cv.namedWindow("input")
cv.moveWindow('input', 0, 0)

region = ROI("input")


for key, frame in autoStream():

    frame = cv.resize(frame, (1080,940))
    
    if region.roi:
        [x1,y1,x2,y2] = region.roi
        trozo = frame[y1:y2+1, x1:x2+1]
        trozo = cv.cvtColor(trozo,cv.COLOR_BGR2GRAY).astype(float)/255
        putText(frame, str(round(np.mean(trozo),2)), orig=(x1,y2+15))
        if key == ord('c'):
            cv.imshow("trozo", trozo)
        if key == ord('x'):
            region.roi = []

        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        putText(frame, f'{x2-x1+1}x{y2-y1+1}', orig=(x1,y1-8))

   

    h,w,_ = frame.shape
    putText(frame, f'{w}x{h}')
    cv.imshow('input',frame)