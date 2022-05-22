#!/usr/bin/env python

# Calculamos y mostramos los puntos SIFT

import cv2 as cv
import time
import argparse
from umucv.stream import autoStream
from umucv.util import putText

# inicializamos el detector con los parámetros de trabajo deseados
# mira en la documentación su significado y prueba distintos valores
# https://docs.opencv.org/3.4/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html

parser = argparse.ArgumentParser(description='Meeeh')
parser.add_argument('-im', metavar='--image', type=str, required = True,
                        help='path to image')
args = parser.parse_args()

sift = cv.SIFT_create(nfeatures=700, contrastThreshold=0.1, edgeThreshold=8)
# sift = cv.AKAZE_create()
img = cv.resize(cv.imread(args.im),(480,640))
cv.namedWindow("image")
if type(img) == type(None):
    print('Error al cargar la imagen')
else:
    
    while True:
        frame = img.copy()
        t0 = time.time()
        # invocamos al detector (por ahora no usamos los descriptores)
        keypoints , _ = sift.detectAndCompute(frame, mask=None)
        t1 = time.time()

        putText(frame, '{} keypoints  {:.0f} ms'.format(len(keypoints), 1000*(t1-t0)))

        # dibujamos los puntos encontrados, con un círculo que indica su tamaño y un radio
        # que indica su orientación.
        # al mover la cámara el tamaño y orientación debe mantenerse coherente con la imagen
        flag = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        cv.drawKeypoints(frame, keypoints, frame, color=(100,150,255), flags=flag)
        
        cv.imshow('image', frame)
        if cv.waitKey(0) == 27: 
            break