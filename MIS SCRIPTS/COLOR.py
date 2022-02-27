import argparse
import cv2               as     cv
import numpy             as     np
import matplotlib.pyplot as plt
from   umucv.util        import ROI
from   collections       import deque
from   sklearn.metrics   import mean_squared_error


def readrgb(file):
    return cv.cvtColor( cv.resize(cv.imread(file), (1000,500)), cv.COLOR_BGR2RGB)

def calNormHistRGB(img):
    '''Dada una imagen RGB devuelve las frecuencias normalizadas de cada valor de pixel para cada canal de color '''
    hR,_ = np.histogram(img[:,:,0].flatten(), np.arange(0, 256)) 
    hG,_ = np.histogram(img[:,:,1].flatten(), np.arange(0, 256))
    hB,_ = np.histogram(img[:,:,2].flatten(), np.arange(0, 256))
    return hR/np.sum(hR), hG/np.sum(hG), hB/np.sum(hB)

parser = argparse.ArgumentParser(description='Clasificación de imagen en base a ROIs como modelos')
parser.add_argument('-im', metavar='--image', type=str, required=True,
                    help='La imagen con la que tratar')

args = parser.parse_args()
print(vars(args))


cv.namedWindow("image")
cv.moveWindow("image", 0, 0)

cv.namedWindow("models")
cv.moveWindow("models", 0, 530)

cv.namedWindow("detected")
cv.moveWindow("detected", 1030,0)

region = ROI("image")

# original_img se trata de la imagen original a la que no le afecta ningun cambio (p.e, las lineas amarillas de un ROI)
img = readrgb(args.im)
original_img = img.copy()
models = deque(maxlen = 3)
histograms = deque(maxlen = 3) 
res_models = deque(maxlen = 3)
detected = []
cont = True

while cont:
    if region.roi:
        [x1,y1,x2,y2] = region.roi
        img = readrgb(args.im)
        cv.rectangle(img, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)

    key = cv.waitKey(1)

    # Capturar el ROI = guardar el modelo
    if key == ord('c') or key == ord('C'):
        model = original_img[y1:y2+1, x1:x2+1]
        # Guardamos el modelo original
        models.append(model)
        # Lista de modelos con tamaño ajustado para mostrarlos en pantalla
        res_models.append(cv.resize(model, (333,300)))
        # En la lista de histogramas guardamos los histogramas de cada canal de una imagen como una tupla (R,G,B)
        histograms.append((calNormHistRGB(model)))


    # Detectar el ROI = seleccionar el modelo mas parecido al ROI
    # Tecla espacio
    if key == 32:
        detected.clear()
        piece = original_img[y1:y2+1, x1:x2+1]
        R,G,B = calNormHistRGB(piece)
        rgbConcat = np.concatenate((R.flatten(),G.flatten(),B.flatten()))

        mseList = []
        it = 1
        for hR,hG,hB in histograms:
            histContat = np.concatenate((hR.flatten(),hG.flatten(),hB.flatten()))
            mse = mean_squared_error(rgbConcat, histContat)
            mseList.append(mse)
            print("MSE con el modelo " + str(it) +" = " + str(mse))
            it += 1

        detected.append(models[np.argmin(mseList)])


    # Tecla ESC
    # Salir
    if key == 27: 
        cont = False

    if (models):
        cv.imshow("models",np.hstack(res_models))

    cv.imshow("image",img)

    if (detected):
        cv.imshow("detected",detected[0])

cv.destroyAllWindows()