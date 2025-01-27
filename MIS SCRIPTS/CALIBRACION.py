#!/usr/bin/env python

import argparse
import cv2         as cv
import numpy       as np
from   umucv.util  import putText
from   collections import deque


parser = argparse.ArgumentParser(description='Mide la distancia en pixels entre 2 puntos seleccionados')
parser.add_argument('-im', metavar='--image', type=str, required=True,
                    help='La imagen con la que tratar')
parser.add_argument('-f', metavar='--focalLength', type=float, required=False, default = 4058.625,
                    help='La distancia focal medida en pixels (por defecto, 4058.625')
args = parser.parse_args()
print(vars(args))


def readrgb(file):
    return cv.cvtColor( cv.resize(cv.imread(file), (3468,4624)), cv.COLOR_BGR2RGB)

def manejador(event, x, y, flags, param):
    global points, modified
    if event == cv.EVENT_LBUTTONDOWN:
        print(x,y) 
        points.append((x,y))
    elif event == cv.EVENT_RBUTTONDOWN:
        points.clear()
        print("CLEARED")
        global image, original_image
        image = original_image.copy()
    
    modified = True


def drawCircle(points, frame):
    for p in points:
        cv.circle(frame,p, 10, (0,0,255), -1)


def calcAngle(p1,p2,f,frame):
    h,w,_ = frame.shape
    u = [abs(p1[0]) - w/2, abs(p1[1]) - h/2, f]
    v = [abs(p2[0]) - w/2, abs(p2[1]) - h/2, f]

    return np.degrees(abs(np.arccos(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)))))

def drawDistancesAndAngles(points,frame):
    l = list(points)
    for (p1,p2) in zip(l,l[1:]):
        cv.line(frame,p1,p2,(0,255,0))
        x1,y1 = p1
        x2,y2 = p2
        dist = round((abs(x2**2-x1**2) + abs(y2**2-y1**2))**0.5,2) #Redondeamos a 2 decimales
        mid = ((x1+x2)//2, (y1+y2)//2)
        alpha = calcAngle(p1,p2,args.f,frame)
        putText(frame,str(dist)+' pixels' + ', alpha = ' + str(alpha),mid)


cv.namedWindow("image")
cv.setMouseCallback("image", manejador)


points = deque([], maxlen = 2)
clicked = False
image = readrgb(args.im)
original_image = image.copy()
print(image.shape)
modified = True
cont = True
while cont:

    # help.show_if(key, ord('h'))
    key = cv.waitKey(1)

    if key == 27:
        cont = False
    
    elif modified:
        image = original_image.copy();
        drawCircle(points,image)
        drawDistancesAndAngles(points,image)
        # No dibujamos la imagen a no ser que haya cambios o sea la primera vez que la mostramos
        cv.imshow('image',image)
        modified = False

cv.destroyAllWindows()