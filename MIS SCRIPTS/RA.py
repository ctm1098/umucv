import argparse
import cv2          as cv
import numpy        as np
from umucv.stream   import Camera
from umucv.htrans   import htrans, Pose
from umucv.contours import extractContours, redu
from umucv.util     import lineType, cube


points = []
tam=None
WH = '640x480'

def manejador(event,x,y,flags,param):
    tamano = cube
    if event == cv.EVENT_LBUTTONDOWN:
        if not points:
            tam*=2
            points.append(tam)
        else:
            tam = points[0]*2
            points[0] = tam
    if event == cv.EVENT_RBUTTONDOWN:
        if not points:
            tam = tamano/2
            points.append(tam)
        else:
            tam = points[0]/2
            points[0] = tam
            


def polygons(cs,n,prec=2):
    rs = [ redu(c,prec) for c in cs ]
    return [ r for r in rs if len(r) == n ]

def rots(c):
    return [np.roll(c,k,0) for k in range(len(c))]

def Kfov(sz,hfovd):
    hfov = np.radians(hfovd)
    f = 1/np.tan(hfov/2)
    w,h = sz
    w2 = w / 2
    h2 = h / 2
    return np.array([[f*w2, 0,    w2],
                     [0,    f*w2, h2],
                     [0,    0,    1 ]])

def bestPose(K,view,model):
    poses = [ Pose(K, v.astype(float), model) for v in rots(view) ]
    return sorted(poses,key=lambda p: p.rms)[0]


parser = argparse.ArgumentParser(description='Aplicación de realidad aumentada')
parser.add_argument('-resize',  metavar='--resizeFrame', type = str, required = False, default = WH, help='Ancho y alto del frame')
args = parser.parse_args()

W,H = (int(args.resize.split('x')[0]), int(args.resize.split('x')[1]))
if W <= 0 or H <= 0:
    print('Error with -resize argument')
    print('Frame dimensions must be positive integers')
    exit()


K = Kfov((W,H), 60)

marker = np.array(
       [[0,   0,   0],
        [0,   1,   0],
        [0.5, 1,   0],
        [0.5, 0.5, 0],
        [1,   0.5, 0],
        [1,   0,   0]])

square = np.array(
       [[0,   0,   0],
        [0,   1,   0],
        [1,   1,   0],
        [1,   0,   0]])
    

color = (0,0,0)
cv.namedWindow('ra')
dicCol = {ord('r'): (0,0,255), ord('g'): (0,255,0), ord('b'): (255,0,0)}
tam = cube
cam = Camera(size = (W,H))

while True:
    frame = cam.frame.copy()
    
    key = cv.waitKey(1) & 0xFF

    if key == 27: 
        break

    if dicCol.get(key) is not None:
        color = dicCol[key]
    elif key == ord('+'):
        tam*=2
    elif key == ord('-'):
        tam = tam/2
    
    g = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cs = extractContours(g, minarea=5, reduprec=2)
    good = polygons(cs,6,3)
    poses = []
    for g in good:
        p = bestPose(K,g,marker)
        if p.rms < 2:
            poses += [p.M]

    cv.drawContours(frame,[htrans(M,tam).astype(int) for M in poses], -1, color, 2, lineType)  
    cv.imshow("ra",frame)

cam.stop() 
cv.destroyAllWindows()