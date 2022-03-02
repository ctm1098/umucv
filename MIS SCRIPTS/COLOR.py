import argparse
import cv2               as     cv
import numpy             as     np
import matplotlib.pyplot as plt
from   umucv.util        import ROI
from   collections       import deque
from   sklearn.metrics   import mean_squared_error


def readrgb(file):
    return cv.cvtColor( cv.resize(cv.imread(file), (1000,500)), cv.COLOR_BGR2RGB)

def calcNormHistRGB(img):
    '''Dada una imagen RGB devuelve las frecuencias normalizadas de cada valor de pixel para cada canal de color '''
    hR,_ = np.histogram(img[:,:,0], np.arange(0, 256)) 
    hG,_ = np.histogram(img[:,:,1], np.arange(0, 256))
    hB,_ = np.histogram(img[:,:,2], np.arange(0, 256))
    return hR/np.sum(hR), hG/np.sum(hG), hB/np.sum(hB)

def adjustROI(im):
    return cv.resize(im, (333,300))

parser = argparse.ArgumentParser(description='Clasificación de imagen en base a ROIs como modelos')
parser.add_argument('-im', metavar='--image', type=str, required=True,
                    help='La imagen con la que tratar')
parser.add_argument('-models', metavar='--models_count', type=int, default=3,
                    help='El máximo número de modelos permitidos (por defecto, 3)')

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
histograms = deque(maxlen = 3) 
models = deque(maxlen = 3)
detected = deque(maxlen = 1)
prevDims = []
cont = True

while cont:
    if region.roi:
        [x1,y1,x2,y2] = region.roi
        # prevDims son las dimensiones del ROI en iteracion previa, de esta forma nos evitamos leer y dibujar el ROI mas veces de las necesarias
        if (prevDims != [x1,y1,x2,y2]):
            prevDims = [x1,y1,x2,y2]
            img = original_img.copy()
            cv.rectangle(img, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)

    key = cv.waitKey(1)

    # Capturar el ROI = guardar el modelo
    if key == ord('c') or key == ord('C') and prevDims:
        # Modelos con tamaño ajustado para mostrarlos en pantalla
        model = adjustROI(original_img[y1:y2+1, x1:x2+1])
        models.append(model)
        # En la lista de histogramas guardamos los histogramas de cada canal de una imagen como una tupla (R,G,B)
        histograms.append((calcNormHistRGB(model)))


    # Seleccionar el modelo mas parecido al ROI indicado
    # Tecla espacio
    if key == 32 and models:
        """ piece = adjustROI(original_img[y1:y2+1, x1:x2+1])
        R,G,B = calcNormHistRGB(piece)
        rgbConcat = np.concatenate((R,G,B))

        mseList = []
        it = 1
        for hR,hG,hB in histograms:
            histContat = np.concatenate((hR,hG,hB))
            mse = mean_squared_error(rgbConcat, histContat)
            mseList.append(mse)
            print("MSE con el modelo " + str(it) +" = " + str(mse))
            it += 1 """

        mseList = [mean_squared_error(np.concatenate((calcNormHistRGB(original_img[y1:y2+1, x1:x2+1]))), np.concatenate((hR,hG,hB))) for (hR,hG,hB) in histograms]

        detected.append(models[np.argmin(mseList)])


    # Tecla ESC
    # Salir
    if key == 27: 
        cont = False

    if (models):
        cv.imshow("models",np.hstack(models))

    cv.imshow("image",img)

    if (detected):
        cv.imshow("detected",detected[0])

cv.destroyAllWindows()