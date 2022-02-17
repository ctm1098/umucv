import numpy as np
import cv2 as cv

from umucv.util import ROI, putText
from umucv.stream import autoStream

from umucv.util import Help

help = Help(
"""
HELP WINDOW

Press left-click to create a ROI

C: captures the ROI and show the difference between capture and live ROI 
X: deletes the ROI
ESC: exit

h: show/hide help
""")

cv.namedWindow("input")
cv.moveWindow('input', 0, 0)

region = ROI("input")
captured = False
#Dimensiones previas de un ROI
prevDim = []
for key, frame in autoStream():

    help.show_if(key, ord('h'))
    
    if region.roi:
        #Recogemos datos del ROI (puntos del rectangulo y region de la imagen)
        [x1,y1,x2,y2] = region.roi
        
        #Si estamos creando un nuevo ROI distinto del anterior...
        if (prevDim and [x1,y1,x2,y2] != prevDim):
            #Si ya habiamos capturado uno, eliminamos las ventanas (si no, dara errores)
            if (captured):
                cv.destroyWindow("trozo")
                cv.destroyWindow("diferencia")
            captured = False

        liveROI = frame[y1:y2+1, x1:x2+1]        
        if key == ord('c'):
            #Si se captura, guardamos el estado del ROI en ese instante en staticROI y activamos el flag
            captured = True
            staticROI = frame[y1:y2+1, x1:x2+1]
            cv.imshow("trozo", staticROI)
        
        if key == ord('x'):
            #Si eliminamos el ROI, desactivamos el flag
            #Si se capturo un ROI previamente, destruimos el resto de ventanas
            if (captured):
                cv.destroyWindow("trozo")
                cv.destroyWindow("diferencia")
            region.roi = []
            captured = False
            

        #Dibujamos el ROI en la imagen y sus dimensiones
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        putText(frame, f'{x2-x1+1}x{y2-y1+1}', orig=(x1,y1-8))
        
        #Si hemos capturado el ROI, mostramos la "imagen diferencia" entre el staticROI y el ROI activo del frame
        #MD = Mean Difference
        if(captured):
            diff = cv.absdiff(liveROI,staticROI)
            cv.imshow("diferencia",diff)
            putText(frame, "MD =" + str(round(np.mean(diff),2)), orig=(x2-50,y2+15))

        #Guardamos las dimensiones de este ROI para compararlas en el siguiente frame
        prevDim = [x1,y1,x2,y2]

    h,w,_ = frame.shape
    putText(frame, f'{w}x{h}')
    cv.imshow('input',frame)