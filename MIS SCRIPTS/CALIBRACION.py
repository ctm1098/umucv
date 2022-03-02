#!/usr/bin/env python

import argparse
import cv2 as cv
import numpy as np
from umucv.util   import   putText
from collections  import  deque
from umucv.util import Help
from math       import pi


parser = argparse.ArgumentParser(description='Mide la distancia en pixels entre 2 puntos seleccionados')
parser.add_argument('-im', metavar='--image', type=str, required=True,
                    help='La imagen con la que tratar')
parser.add_argument('-f', metavar='--focalLength', type=float, required=False, default = 775.5211,
                    help='La distancia focal medida en pixels')
args = parser.parse_args()
print(vars(args))


def readrgb(file):
    return cv.cvtColor( cv.resize(cv.imread(file), (1000,500)), cv.COLOR_BGR2RGB)

def manejador(event, x, y, flags, param):
    global points, clicked
    if event == cv.EVENT_LBUTTONDOWN:
        print(x,y) 
        points.append((x,y))
        clicked = True
    elif event == cv.EVENT_RBUTTONDOWN:
        points.clear()
        print("CLEARED")
        clicked = False
        global image, original_image
        image = original_image.copy()


def drawCircle(points, frame):
    for p in points:
        cv.circle(frame,p, 10, (0,0,255), -1)

def drawDistancesAndAngles(points,frame):
    l = list(points)
    for (p1,p2) in zip(l,l[1:]):
        cv.line(frame,p1,p2,(0,255,0))
        x1,y1 = p1
        x2,y2 = p2
        dist = round((abs(x2**2-x1**2) + abs(y2**2-y1**2))**0.5,2) #Redondeamos a 2 decimales
        mid = ((x1+x2)//2, (y1+y2)//2)
        alpha = round(abs(np.arctan(dist/args.f) * 180 / pi),2)
        putText(frame,str(dist)+' pixels' + ', alpha = ' + str(alpha),mid)


cv.namedWindow("image")
cv.setMouseCallback("image", manejador)


points = deque([], maxlen = 2)
clicked = False
image = readrgb(args.im)
original_image = image.copy()
print(image.shape)
cont = True
modified = True

while cont:

    # help.show_if(key, ord('h'))
    key = cv.waitKey(1)

    if key == 27:
        cont = False
    
    else:
        # Con esta comprobacion evitamos dibujar los mismos circulos mas de una vez
        if clicked:
            image = original_image.copy();
            drawCircle(points,image)
            drawDistancesAndAngles(points,image)
            drawDistancesAndAngles(points,image)
            modified = True
            clicked = False    

        # No dibujamos la imagen a no ser que haya cambios o sea la primera vez que la mostramos
        if (modified):
            cv.imshow('image',image)
            modified = False

cv.destroyAllWindows()