from collections import deque
from cv2 import cv2 as cv
import numpy as np
import argparse
from umucv.util import putText
import matplotlib.pyplot as plt

#Dimensiones de la tarjeta = 8.5 x 5.4 cm


SIZE = (900,880)
H = 5.4
W = 8.5

pts = deque(maxlen = 2)

def shcont(c, color='blue', nodes=True):
    x = c[:,0]
    y = c[:,1]
    x = np.append(x,x[0])
    y = np.append(y,y[0])
    plt.plot(x,y,color)
    if nodes: plt.plot(x,y,'.',color=color, markersize=11)

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

def getInfo(path):
    f = open(path)
    lines = [l.split(":")[1].strip() for l in f.readlines()]
    f.close()
    imFile = lines[0]
    size = (float(lines[1].split('x')[0]), float(lines[1].split('x')[1]))
    refP = [float(p) for p in lines[2].split(' ')]
    realP = [float(p) for p in lines[3].split(' ')]
    #factor
    distP = (abs(refP[0]**2-refP[2]**2) + abs(refP[1]**2-refP[3]**2))**0.5
    factor = size[1]/distP
    return imFile,refP,realP,factor

parser = argparse.ArgumentParser(description='Rectifica la imagen de un plano para medir distancias tomando manualmente referencias conocidas')
parser.add_argument('-fin', metavar='--inputFile', type=str, required = True, help='Path al fichero con los datos de entrada')
args = parser.parse_args()

cv.namedWindow("original")
cv.namedWindow("rectif")
cv.setMouseCallback("original", manejador)


imFile,refP,realP,factor = getInfo(args.fin)
imOriginal = cv.imread(imFile)
fig = plt.figure(figsize=(6,8))
plt.imshow(imOriginal)
fig.show()

ref = np.array([
        [refP[0],refP[1]],
        [refP[2],refP[3]],
        [refP[4],refP[5]],
        [refP[6],refP[7]]])

real = np.array([
        [realP[0],realP[1]],
        [realP[2],realP[3]],
        [realP[4],realP[5]],
        [realP[6],realP[7]]])

nuevo,x = cv.findHomography(ref, real)
rec = cv.warpPerspective(imOriginal, nuevo, SIZE)

shcont(ref,color='red')

while True:

    im = imOriginal.copy()

    for p in pts:
        cv.circle(im, p, 3, (0,0,255), -1)
    if len(pts) == 2:
        x1,y1 = pts[0]
        x2,y2 = pts[1]
        mid = ((x1+x2)//2, (y1+y2)//2)
        cv.line(im, pts[0], pts[1], (0,255,0))
        distP = round((abs(x2**2-x1**2) + abs(y2**2-y1**2))**0.5,2)
        putText(im,'{} cm'.format(distP*factor),mid)

    cv.imshow("original", im)
    cv.imshow("rectif", rec)

    key = cv.waitKey(1)
    if key == 27:
        break

cv.destroyAllWindows()

