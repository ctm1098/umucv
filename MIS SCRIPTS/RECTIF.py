from collections import deque
from cv2 import cv2 as cv
import numpy as np
import argparse
from umucv.util import putText
import matplotlib.pyplot as plt

SIZE = (640,480)

pts = deque(maxlen = 2)

def shcont(c, color='blue', nodes=True):
    x = c[:,0]
    y = c[:,1]
    x = np.append(x,x[0])
    y = np.append(y,y[0])
    plt.plot(x,y,color)
    if nodes: plt.plot(x,y,'.',color=color, markersize=11)

def drawDistances(points,frame,factor):
    l = list(points)
    for (p1,p2) in zip(l,l[1:]):
        cv.line(frame,p1,p2,(0,255,0))
        x1,y1 = p1
        x2,y2 = p2
        dist = round((abs(x2**2-x1**2) + abs(y2**2-y1**2))**0.5,2) * factor #Redondeamos a 2 decimales
        mid = ((x1+x2)//2, (y1+y2)//2)
        putText(frame,str(dist)+' pixels',mid)

def drawCircle(points, frame):
    for p in points:
        cv.circle(frame,p, 10, (0,0,255), -1)

def manejador(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x,y)
        pts.append((x,y))
    elif event == cv.EVENT_RBUTTONDOWN:
        pts.clear()
        print("CLEARED")

'''parser = argparse.ArgumentParser(description='ectifica la imagen de un plano para medir distancias tomando manualmente referencias conocidas')
parser.add_argument('-in', metavar='--inputFile', type=str, required = True, help='Path al fichero con los datos de entrada')
parser.add_argument('-resize',  metavar='--resizeFrame', type = str, required = False, default = SIZE, help='Ancho y alto de la imagen')
args = parser.parse_args()'''


cv.namedWindow("image")
cv.setMouseCallback("image", manejador)

#W,H = (int(args.resize.split('x')[0]), int(args.resize.split('x')[1]))
im = cv.imread('objetos3.jpg')

fig = plt.figure(figsize=(6,8))
#plt.imshow(im)


ref = np.array([
        [694,857],
        [787,921],
        [926,868],
        [830,813]])

distX = (abs(787**2-921**2) + abs(926**2-868**2))**0.5
distY = (abs(694**2-787**2) + abs(857**2-921**2))**0.5
prop = distX/distY

real = np.array([
    [500,500],
    [500,560],
    [594.4,560],
    [594.4,500]])

nuevo,x = cv.findHomography(ref, real)
rec = cv.warpPerspective(im, nuevo, (900,880))

shcont(ref,color='red')

plt.imshow(im)
fig.show()

while True:

    recC = rec.copy()

    for p in pts:
        cv.circle(recC, p, 3, (0,0,255), -1)
    if len(pts) == 2:
        x1,y1 = pts[0]
        x2,y2 = pts[1]
        mid = ((x1+x2)//2, (y1+y2)//2)
        cv.line(recC, pts[0], pts[1], (0,255,0))
        dist = np.linalg.norm( np.array(pts[0]) - pts[1]) * 0.09
        putText(recC,str(dist)+' cm',mid)

    cv.imshow("image", recC)

    key = cv.waitKey(1)
    if key == 27:
        break

cv.destroyAllWindows()

