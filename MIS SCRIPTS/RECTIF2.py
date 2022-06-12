import  argparse
import  numpy               as np
from    umucv.util          import putText
from    collections         import deque
from    umucv.htrans        import htrans
from    umucv.stream        import Camera
from    time                import time
from    cv2                 import cv2 as cv


T_DIST = ['km','hm','dam','m','dm','cm','mm','um','nm']
DEF_DIST = 'cm'
WH = '640x480'

ref = []
ptsDist = deque(maxlen = 2)

def drawCircle(points, frame):
    for p in points:
        cv.circle(frame,p, 10, (0,0,255), -1)

def handler(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        if len(ref) < 5:
            print("Punto para el plano: ({},{})".format(x,y))
            ref.append((x,y))
        else:
            print("Punto para medir distancias: ({},{})".format(x,y))
            ptsDist.append((x,y))

    elif event == cv.EVENT_RBUTTONDOWN:
        if len(ptsDist) > 0:
            ptsDist.clear()
            print("Puntos de distancia borrados")
        elif len(ref) > 0:
            ref.clear()
            print("Puntos de plano borrados")
        

def f(x):
    pass

parser = argparse.ArgumentParser(description='Aplicación que rectifica un plano seleccionado de una imagen para medir distancias entre los objetos')
parser.add_argument('-u', metavar='--lengthUnit', type=str, required = False, default = DEF_DIST,
                     help='Unidad de longitud (por defecto, {})'.format(DEF_DIST))
parser.add_argument('-resize',  metavar='--resizeFrame', type = str, required = False, default = WH, help='Ancho y alto del frame')

args = parser.parse_args()
unit = args.u
if unit not in T_DIST:
    print("Unidad de distancia no reconocida. Se debe usar una de las siguientes: {}".format(T_DIST))
    exit()

W,H = (int(args.resize.split('x')[0]), int(args.resize.split('x')[1]))
if W <= 0 or H <= 0:
    print('Error with -resize argument')
    print('Frame dimensions must be positive integers')
    exit()


cv.namedWindow("original")
cv.setMouseCallback("original", handler)  

cv.createTrackbar('Alto ({})'.format(unit), 'original', 46, 300, f)
cv.createTrackbar('Ancho ({})'.format(unit), 'original', 84, 300, f)


alto = None
ancho = None
created = False
cam = Camera(size = (W,H))

while True:

    frame = cam.frame.copy()
    
    key = cv.waitKey(1) & 0xFF
    
    if key == 27: 
        break
    
    # Dibujamos los puntos del objeto conocido
    for p in ref[:4]:
        cv.circle(frame, p, 3, (0,255,0), -1)
    
    time0 = time()
         
    if len(ref) == 5:
        cv.line(frame, ref[0], ref[1], (0,255,0))
        cv.line(frame, ref[1], ref[2], (0,255,0))
        cv.line(frame, ref[2], ref[3], (0,255,0))
        cv.line(frame, ref[3], ref[0], (0,255,0))
        

        alto = cv.getTrackbarPos('Alto ({})'.format(unit), 'original')
        ancho = cv.getTrackbarPos('Ancho ({})'.format(unit), 'original')
        
        #Punto base
        x, y = ref[4]
        
        real = list([(x,y),(x+ancho, y),(x+ancho, y+alto),(x, y+alto)])
    
        # Calculamos la homografia y la mostramos
        homog , _ = cv.findHomography(np.array(ref[:4]), np.array(real))
        rec = cv.warpPerspective(frame, homog, (W,H))
        if not created:
            created = True
            cv.namedWindow("rectif")
        cv.imshow('rectif',rec)

        
        # Dibujamos los puntos para medir                                        
        for p in ptsDist:
            cv.circle(frame, p, 3, (255,0,0), -1)
        
        if len(ptsDist) == 2:
            (x1, y1), (x2, y2) = ptsDist

            cv.line(frame, (x1,y1), (x2,y2), (255,0,0))
            
            pointsRec = htrans(homog, ptsDist)
            
            dist = np.linalg.norm(np.array(pointsRec[0]) - pointsRec[1])
            
            mid = ((x1+x2)//2, (y1+y2)//2)
            putText(frame, f'{dist:.1f} ' + unit, orig=mid)

    elif created:
        created = False
        cv.destroyWindow("rectif")
    
    time1 = time()
    
    putText(frame, f'{1000*(time1-time0):.0f} ms')
    cv.imshow('original', frame)

cam.stop()
cv.destroyAllWindows()


