import cv2 as cv
from umucv.util   import   putText
import sys
import skimage.io        as io
from collections  import  deque

points = deque([], maxlen = 6)
cv.namedWindow("webcam")

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

frame = cv.imread("images\ctm0.png")
while True:
    drawCircle(points,frame)
    drawDistances(points,frame)
    cv.imshow('webcam',frame)

cv.destroyAllWindows()