import argparse
import cv2               as     cv
import numpy             as     np
from   umucv.util        import ROI, putText
from   collections       import deque
from   sklearn.metrics   import mean_squared_error

parser = argparse.ArgumentParser(description='Clasificación de imagen en base a ROIs como modelos')
parser.add_argument('-im', metavar='--image', type=str, required=True,
                    help='La imagen con la que tratar')
parser.add_argument('-models', metavar='--models_count', type=int, default=3,
                    help='El máximo número de modelos permitidos (por defecto, 3)')
parser.add_argument('-bins', metavar='--image', type=int, required=False, default = 8,
                    help='El numero de bins en los que dividir el rango de valores del hisograma (por defecto, 8)')

args = parser.parse_args()
print(vars(args))

def readrgb(file):
    return cv.cvtColor( cv.resize(cv.imread(file), (1000,500)), cv.COLOR_BGR2RGB)

def calcHistRGB(img):
    '''Dada una imagen RGB devuelve las frecuencias de cada valor de pixel para cada canal de color '''
    R = np.histogram(img[:,:,0], args.bins, range = (0,256)) 
    G = np.histogram(img[:,:,1], args.bins, range = (0,256))
    B = np.histogram(img[:,:,2], args.bins, range = (0,256))
    return R,G,B

def adjustROI(im):
    return cv.resize(im, (333,300))

def adjustLimit(hists, bins, limit, limitW):
    max = np.max([np.max(h) for h in hists])
    t = tuple([h*limit/max for h in hists])
    bins = [b * limitW/255 for b in bins]
    return (bins,t[0]), (bins,t[1]), (bins,t[2])



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
        putText(img, f'{x2-x1+1}x{y2-y1+1}', orig=(x1,y1-8))
        # prevDims son las dimensiones del ROI en iteracion previa, de esta forma nos evitamos leer y dibujar el ROI mas veces de las necesarias
        if (prevDims != [x1,y1,x2,y2]):
            prevDims = [x1,y1,x2,y2]
            img = original_img.copy()
            model = img[y1:y2+1, x1:x2+1]
            hR,hG,hB = calcHistRGB(model)
            print("=========== HR ===============")
            print(hR)
            print("==============================")
            print("=========== HG ===============")
            print(hG)
            print("==============================")
            print("=========== HB ===============")
            print(hB)
            print("==============================")
            hR2, hG2, hB2 = adjustLimit([hR[0],hG[0],hB[0]],hR[1],abs(y1-y2+1), abs(x1-x2+1))
            print(hR2)
            draw_pointsR = np.array(list(zip(hR2[0], hR2[1]))).astype(np.int32)
            draw_pointsG = np.array(list(zip(hG2[0], hG2[1]))).astype(np.int32)
            draw_pointsB = np.array(list(zip(hB2[0], hB2[1]))).astype(np.int32)
            cv.rectangle(img, (x1,y1), (x2,y2), color=(0,255,255), thickness = 2)
            cv.polylines(model, [draw_pointsR], False, (255,0,0), thickness = 2)  # args: image, points, closed, color
            cv.polylines(model, [draw_pointsG], False, (0,255,0), thickness = 2)
            cv.polylines(model, [draw_pointsB], False, (0,0,255), thickness = 2)

    key = cv.waitKey(1)

    # Capturar el ROI = guardar el modelo
    if key == ord('c') or key == ord('C') and prevDims:
        # Modelos con tamaño ajustado para mostrarlos en pantalla
        model = adjustROI(original_img[y1:y2+1, x1:x2+1])
        models.append(model)
        # En la lista de histogramas guardamos los histogramas de cada canal de una imagen como una tupla (R,G,B)
        histograms.append((calcHistRGB(model)))


    # Seleccionar el modelo mas parecido al ROI indicado
    # Tecla espacio
    if key == 32 and models:
        piece = adjustROI(original_img[y1:y2+1, x1:x2+1])
        R,G,B = calcHistRGB(piece)
        rgbConcat = np.concatenate((R,G,B))

        mseList = []
        it = 1
        for hR,hG,hB in histograms:
            histContat = np.concatenate((hR,hG,hB))
            mse = mean_squared_error(rgbConcat, histContat)
            mseList.append(mse)
            print("MSE con el modelo " + str(it) +" = " + str(mse))
            it += 1

        # mseList = [mean_squared_error(np.concatenate((calcNormHistRGB(original_img[y1:y2+1, x1:x2+1]))), np.concatenate((hR,hG,hB))) for (hR,hG,hB) in histograms]

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