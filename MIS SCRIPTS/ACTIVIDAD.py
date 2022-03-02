import argparse
import cv2          as cv
import numpy        as np
from   umucv.stream import autoStream
from   umucv.util   import Video, ROI, putText
from   time         import time

cv.namedWindow("input")
cv.moveWindow('input', 0, 0)

parser = argparse.ArgumentParser(description='Graba y guarda archivos de vídeo de 3 segundos con la webcam en caso de que se detecte en la imagen una anomalía que sobrepase un cierto umbral en el ROI seleccionado')
parser.add_argument('-thres', metavar='--threshold', type=float, default = 10,
                    help='La diferencia media umbral de imagen entre entre 2 ROIs que se debe superar para considerar una anomalía en la imagen (por defecto, 10)')
parser.add_argument('-time', metavar='--videoLength', type=float, default = 3,
                    help='La longitud del vídeo en segundos (por defecto, 3)')
                    
args = parser.parse_args()
variables = vars(args)
print(variables)

region = ROI("input")
captured = False
prevDim = []
start = video = 0

def startVideo():
    '''Inicia el proceso de grabacion y devuelve el objeto video y la marca de tiempo de inicio'''
    print("\nAnomaly detected with "+str(round(diff,2))+" mean difference")
    video = Video(fps=30, codec="MJPG",ext="avi")
    video.ON = True
    return video, time()

def stopVideo(video,t = 0, info = 0):
    '''Detiene un video y devuelve valores nulos para este y la marca de inicio de grabacion.
    Puede recibir como argumento adicional el tiempo de grabacion de video'''
    global start
    print(str(round(t if t else time()-start, 2)) + " seconds of anomaly recorded")
    if (info):
        print(info)
    video.release()
    return 0,0


for key, frame in autoStream():
    if region.roi:
        #Recogemos datos del ROI (puntos del rectangulo)
        [x1,y1,x2,y2] = region.roi
        
        #Si estamos creando un nuevo ROI distinto del anterior ya capturado...
        if (prevDim and [x1,y1,x2,y2] != prevDim and captured):
            captured = False
            #Si estamos grabando un video con un ROI antiguo...
            if (video):
                video, start = stopVideo(video,info = "A new ROI is beeing created")

        #ROI en directo (se actualiza en cada frame)
        liveROI = frame[y1:y2+1, x1:x2+1]

        if key == ord('c') or key == ord('C'):
            #Si se captura, guardamos el estado del ROI en ese instante en staticROI y activamos el flag
            captured = True
            #ROI estatico (solo guarda la imagen del ROI en el instante de la captura)
            staticROI = frame[y1:y2+1, x1:x2+1]
        
        if key == ord('x') or key == ord('X'):
            #Si eliminamos el ROI, desactivamos el flag
            region.roi = []
            captured = False
            #Si estabamos grabando, detenemos el video
            if (video):
                video, start = stopVideo(video, info = "'X' key pressed, stopping capture")

        #Dibujamos el ROI en la imagen y sus dimensiones
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        putText(frame, f'{x2-x1+1}x{y2-y1+1}', orig=(x1,y1-8))
        
        
        # Graba siempre un maximo de TIME segundos de anomalia
        # Si la anomalia dura mas, para la grabacion y comienza otra
        if(captured):
            #Calculo de la diferencia media
            putText(frame, "Watching...", orig=(x1,y2+15))
            imgDiff = cv.absdiff(liveROI,staticROI)
            diff = np.mean(imgDiff)
            #MD = Mean Difference
            putText(frame, "MD =" + str(round(diff,2)), orig=(x1,y2+30))

            #Si hay una anomalia...
            if (diff >= args.thres):
                putText(frame, "ANOMALY DETECTED", orig=(x1,y2+45))
                #Si ya se estaba grabando un video (video != 0) y han pasado 3 segundos...
                if (video and time() - start >= args.time):
                    video, start = stopVideo(video,args.time)
                #Si no se esta grabando (start == 0)
                else:
                    if (video == 0):
                        video, start = startVideo()
                    else:
                        video.write(frame)
            
            #Si no hay anomalias Y hay un video en marcha...
            elif (video):
                video, start = stopVideo(video, info = "Anomaly gone")
                
        #Guardamos las dimensiones de este ROI para compararlas en el siguiente frame
        prevDim = [x1,y1,x2,y2]

    h,w,_ = frame.shape
    putText(frame, f'{w}x{h}')
    cv.imshow('input',frame)

cv.destroyAllWindows()
if (video):
    stopVideo(video, info = "Exiting program")

