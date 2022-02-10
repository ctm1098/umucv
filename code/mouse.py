#!/usr/bin/env python

import cv2 as cv
from umucv.stream import autoStream
from umucv.util   import   putText
from collections  import  deque
from umucv.util import Help

help = Help(
"""
HELP WINDOW

Left-click: draw circle
Right-Click: clear

SPC: pausa
ESC: exit

h: show/hide help
""")

points = deque([], maxlen = 6)

def fun(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x,y)
        points.append((x,y))
    elif event == cv.EVENT_RBUTTONDOWN:
        points.clear()
        print("CLEARED")

    
cv.namedWindow("webcam")
cv.setMouseCallback("webcam", fun)


def drawCircle(points, frame):
    for p in points:
        cv.circle(frame,p, 10, (0,0,255), -1)

def drawDistances(points,frame):
    l = list(points)
    for (p1,p2) in zip(l,l[1:]):
        cv.line(frame,p1,p2,(0,255,0))
        x1,y1 = p1
        x2,y2 = p2
        dist = round((abs(x2**2-x1**2) + abs(y2**2-y1**2))**0.5,2) #Redondeamos a 2 decimales
        mid = ((x1+x2)//2, (y1+y2)//2)
        putText(frame,str(dist)+' pixels',mid)


for key, frame in autoStream():
    help.show_if(key, ord('h'))
    drawCircle(points,frame)
    drawDistances(points,frame)
    cv.imshow('webcam',frame)

cv.destroyAllWindows()

