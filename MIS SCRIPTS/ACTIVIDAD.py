import cv2          as cv
import numpy        as np
from   umucv.stream import autoStream
from   umucv.util   import Video, ROI, putText
from   time         import time

THRESSHOLD = 10
TIME = 3

cv.namedWindow("input")
cv.moveWindow('input', 0, 0)


region = ROI("input")
captured = False
prevDim = []
video = start = 0
for key, frame in autoStream():
    if region.roi:
        #Recogemos datos del ROI (puntos del rectangulo y region de la imagen)
        [x1,y1,x2,y2] = region.roi
        
        #Si estamos creando un nuevo ROI distinto del anterior...
        if (prevDim and [x1,y1,x2,y2] != prevDim):
            captured = False

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
            if (diff >= THRESSHOLD):
                #Si ya se estaba grabando un video (start != 0) y han pasado 3 segundos...
                if (start != 0 and time() - start >= TIME):
                    video.release()
                    start = 0
                #Si no se esta grabando (start == 0)
                if (start == 0):
                    video = Video(fps=30, codec="MJPG",ext="avi")
                    video.ON = True
                    start = time()
                
                video.write(frame)
            
            #Si no hay anomalias Y hay un video en marcha...
            elif (start != 0):
                video.release()
                start = 0
                
        #Guardamos las dimensiones de este ROI para compararlas en el siguiente frame
        prevDim = [x1,y1,x2,y2]

    h,w,_ = frame.shape
    putText(frame, f'{w}x{h}')
    cv.imshow('input',frame)

cv.destroyAllWindows()
video.release()

