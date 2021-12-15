import cv2
from prediccion import Prediccion

clases = [ "bisturi","cepillo","llaves","maquina","tijera"]

ancho = 256
alto = 256

miModeloCNN = Prediccion("models/modeloD.h5", ancho, alto)
imagen = cv2.imread("dataset/test/tijera/tijera (35).jpg")
claseResultado = miModeloCNN.predecir(imagen)
print("el valor es", clases[claseResultado])