import argparse
import cv2        as cv
import numpy      as np
from glob         import glob
from umucv.stream import Camera
from umucv.util   import putText
from math         import floor

NFEATURES = 500
WH = '640x480'
THRES = 7
MODELS_PATH = './SIFT-models/*.jpg'

def fillborders(frame,model,x_offset,y_offset,col):
    frame[y_offset:y_offset+model.shape[0],0:x_offset] = col
    frame[y_offset:y_offset+model.shape[0],x_offset+model.shape[1]:2*x_offset+model.shape[1]] = col
    frame[y_offset-x_offset:y_offset,0:2*x_offset+model.shape[1]] = col
    frame[y_offset+model.shape[0]:y_offset+x_offset+model.shape[0],0:2*x_offset+model.shape[1]] = col


parser = argparse.ArgumentParser(description='Aplicación de reconocimiento de objetos con la webcam basada en el número de concidencias de keypoints')
parser.add_argument('-nf', metavar='--nfeatures', type=int, required = False, default = NFEATURES, help='Número máximo de keypoints a detectar')
parser.add_argument('-resize',  metavar='--resizeFrame', type = str, required = False, default = WH, help='Ancho y alto del frame')
parser.add_argument('-thres',  metavar='--threshold', type = int, required = False, default = THRES, help='Umbral de coincidencia')
parser.add_argument('-p',  metavar='--pathToModels', type = str, required = False, default = MODELS_PATH, help='Path de la carpeta que contiene los modelos en jpg')
args = parser.parse_args()

# Ancho y alto del frame (el mismo para los modelos)
W,H = (int(args.resize.split('x')[0]), int(args.resize.split('x')[1]))

if W <= 0 or H <= 0:
    print('Error with -resize argument')
    print('Frame dimensions must be positive integers')
    exit()

# Ancho y alto del mini-modelo que solaparemos en el frame original cuando se detecte
modelW,modelH = (floor(W/4.18), floor(H/3.42))
# Offsets para el mini-modelo para solaparlo en el frame
x_offset=10
y_offset=40

cv.namedWindow("camera")
cam = Camera(size = (W,H))

# Parámetros del SIFT
sift = cv.xfeatures2d.SIFT_create(nfeatures=args.nf)
match = cv.BFMatcher()

idx = 0
models = []
files = glob(args.p)
for f in files:
    name = f.split('\\')[-1].split('.')[0].replace('_',' ')
    model = cv.resize(cv.imread(f),(W,H))
    kpts, descs = sift.detectAndCompute(model,mask=None)
    models.append((name,model,kpts,descs))

while True:
    frame = cam.frame.copy()
    
    key = cv.waitKey(1) & 0xFF
    
    if key == 27: 
        break

    if key == ord('x'):
        models.clear()

    if key in [ord('c'), ord('C')]:
        model = frame.copy()
        kpts, descs = sift.detectAndCompute(model,mask=None)
        models.append(("Model_{}".format(idx),model,kpts,descs))
        idx+=1
        print("Added model {}".format(idx))


    if models:
        #and key in [ord('e'), ord('E')]
            
        frame_kpts, frame_descs = sift.detectAndCompute(frame, mask=None)

        modelDetected = np.zeros((modelW,modelW,3),dtype = "uint8")
        best_name = 'None'
        max = 0
        for (name,model,model_kpts,model_descs) in models:
            pts = []
          
            matches = match.knnMatch(frame_descs,model_descs,k=2)
            
            for m in matches:
                if len(m) >= 2:
                    best,second = m
                    if best.distance < 0.85*second.distance:
                        pts.append(best)

            percent = 100*len(pts)/len(model_kpts)

            if (percent > max):
                bestModel = (name,model)
                max = percent
        
        if (percent > args.thres):
            (name,model) = bestModel
            modelDetected = cv.resize(model.copy(), (modelW,modelW))
            best_name = name

        putText(frame, f'{len(frame_kpts)} pts')
        fillborders(frame,modelDetected,x_offset,y_offset,col = (192,192,192))
        frame[y_offset:y_offset+modelDetected.shape[0], x_offset:x_offset+modelDetected.shape[1]] = modelDetected
        putText(frame, f'{percent:.2f}% likelihood', orig = (x_offset,y_offset+modelDetected.shape[0]+30))
        putText(frame, f'{best_name}', orig = (x_offset,y_offset+modelDetected.shape[0]+60))

    #flag = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    #cv.drawKeypoints(frame, frame_kpts, frame, color=(100,150,255), flags=flag)
    cv.imshow('camera',frame)

cam.stop()
cv.destroyAllWindows()

