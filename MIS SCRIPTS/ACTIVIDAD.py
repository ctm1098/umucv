import argparse
import cv2          as cv
import numpy        as np
from   umucv.stream import autoStream
from   umucv.util   import Video, ROI, putText
from   time         import time

cv.namedWindow("input")
cv.moveWindow('input', 0, 0)

parser = argparse.ArgumentParser(description='Graba y guarda archivos de vídeo de 3 segundos con la webcam en caso de que se detecte en la imagen una anomalía que sobrepase un cierto umbral en el ROI seleccionado')
parser.add_argument('-THRES', metavar='--threshold', type=float, default = 10,
                    help='La diferencia media umbral entre el ROI original y el ROI en directo que se debe superar para considerar una anomalía en la imagen (por defecto, 10)')
parser.add_argument('-TIME', metavar='--videoLength', type=float, default = 3,
                    help='La longitud del vídeo en segundos (por defecto, 3)')

args = parser.parse_args()
args = parser.parse_args()
variables = vars(args)
print(variables)

region = ROI("input")
captured = False
prevDim = []
video = start = 0
for key, frame in autoStream():
    if region.roi:
        #Recogemos datos del ROI (puntos del rectangulo y region de la imagen)
        [x1,y1,x2,y2] = region.roi
        
        #Si estamos creando un nuevo ROI distinto del anterior...
        if (prevDim and [x1,y1,x2,y2] != prevDim and captured):
            captured = False
            #Si estamos grabando un video con un ROI antiguo...
            if (start != 0):
                video.release()


        liveROI = frame[y1:y2+1, x1:x2+1]        
        if key == ord('c'):
            #Si se captura, guardamos el estado del ROI en ese instante en staticROI y activamos el flag
            captured = True
            staticROI = frame[y1:y2+1, x1:x2+1]
        
        if key == ord('x'):
            #Si eliminamos el ROI, desactivamos el flag
            #Si se capturo un ROI previamente, destruimos el resto de ventanas
            region.roi = []
            captured = False
            video.release()

        #Dibujamos el ROI en la imagen y sus dimensiones
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        putText(frame, f'{x2-x1+1}x{y2-y1+1}', orig=(x1,y1-8))
        
        
        # Graba siempre un maximo de TIME segundos de anomalia
        # Si la anomalia dura mas, se cierra el video y empieza uno nuevo
        if(captured):
            putText(frame, "Watching...", orig=(x1,y2+15))
            imgDiff = cv.absdiff(liveROI,staticROI)
            diff = np.mean(imgDiff)
            #MD = Mean Difference
            putText(frame, "MD =" + str(round(diff,2)), orig=(x1,y2+30))
            #Si hay una anomalia...
            if (diff >= args.THRES):
                putText(frame, "ANOMALY DETECTED", orig=(x1,y2+45))
                #Si ya se estaba grabando un video (start) y han pasado 3 segundos...
                if (start and time() - start >= args.TIME):
                    video.release()
                    start = 0
                #Si no se esta grabando (start == 0)
                if (start == 0):
                    print("Anomaly detected with "+str(diff)+" mean difference")
                    video = Video(fps=30, codec="MJPG",ext="avi")
                    video.ON = True
                    start = time()
                
                video.write(frame)
            
            #Si no hay anomalias Y hay un video en marcha...
            elif (start):
                video.release()
                start = 0
                
        #Guardamos las dimensiones de este ROI para compararlas en el siguiente frame
        prevDim = [x1,y1,x2,y2]

    h,w,_ = frame.shape
    putText(frame, f'{w}x{h}')
    cv.imshow('input',frame)

cv.destroyAllWindows()
video.release()

