import tensorflow as tf
import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
#componentes de la red neuronal

from keras.models import Sequential
from keras.layers import InputLayer, Input, Conv2D, MaxPool2D, Reshape, Dense, Flatten
#######################################################################

def cargarDatos(rutaOrigen, clases, numeroCategorias, limite, width, height):
    imagenesCargadas = []
    valorEsperado = []
    for categoria in range (0, numeroCategorias):
        for idImagen in range(1, limite[categoria]):
            ruta = rutaOrigen + str(clases[categoria]) + "/" + str(clases[categoria]) + " ({})".format(idImagen) + ".jpg"
            imagen = cv2.imread(ruta)
            imagen = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
            imagen = cv2.resize(imagen, (width, height))
            imagen = imagen.flatten()
            imagen = imagen / 255
            imagenesCargadas.append(imagen)
            probabilidades = np.zeros(numeroCategorias)
            probabilidades[categoria] = 1
            valorEsperado.append(probabilidades)
    imagenesEntrenamiento = np.array(imagenesCargadas)
    valoresEsperados = np.array(valorEsperado)
    return imagenesEntrenamiento, valoresEsperados

########################################################################

ancho = 256
alto = 256
pixeles = ancho * alto
# Imagen RGB -->
numero_canales = 1
forma_imagen = (ancho, alto, numero_canales)
numero_clases = 5
clases = [ "bisturi","cepillo","llaves","maquina","tijera"]
cantidadDatosEntrenamiento = [104, 106,103,143,103]
cantidadDatosPruebas = [104,106,55,143, 55]

################### cargar imagenes

imagenes, probabilidades = cargarDatos("dataset/train/", clases, numero_clases, cantidadDatosEntrenamiento, ancho, alto)

model = Sequential()
#capas de entradas
model.add(InputLayer(input_shape= (pixeles,)))
model.add(Reshape(forma_imagen))


#capas ocultas
#capas convolucionales

model.add(Conv2D(kernel_size=5,strides=2,filters=16,padding="same",activation="selu",name="capa_1"))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(kernel_size=3,strides=1,filters=32,padding="same",activation="selu",name="capa_2"))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(kernel_size=5, strides=2, filters=64, padding="same", activation="selu", name="capa_3"))
model.add(MaxPool2D(pool_size=2, strides=2))

#aplanamiento
model.add(Flatten())
model.add(Dense(128,activation="relu"))

#salida
model.add(Dense(numero_clases,activation="softmax"))

#traducir keras a tensorflow

model.compile(optimizer="adam",loss="categorical_crossentropy", metrics=["accuracy"])

#entrenamiento
model.fit(x=imagenes, y=probabilidades, epochs=7,batch_size=30)

#prueba de modelo

imagenesPrueba, probabilidadesPruebas = cargarDatos("dataset/test/", clases, numero_clases, cantidadDatosPruebas, ancho, alto)
resultados = model.evaluate(x=imagenesPrueba, y=probabilidadesPruebas)
#print("accuracy=", resultados[1])
print("___",resultados)
print("Metricas",model.metrics_names)
print("Accuracy=",resultados)
#Guardar modelo
ruta = "models/modeloD.h5"
model.save(ruta)

#informe de estructura de la red
model.summary()


predicciones = model.predict(imagenesPrueba, batch_size=20, verbose=1)
valorPredicciones = np.argmax(predicciones, axis=1)

#Matriz de confusión
matriz = confusion_matrix(np.argmax(probabilidadesPruebas, axis=1), valorPredicciones)

# Visualizar la matriz de confusión
visualizar = pd.DataFrame(matriz, range(5), range(5))
plt.figure(figsize=(10, 8))
sn.set(font_scale=1.2)  # for label size
sn.heatmap(visualizar,cmap="Greens", annot=True, annot_kws={"size": 10})  # font size
plt.show()

reporte = classification_report(np.argmax(probabilidadesPruebas, axis=1), valorPredicciones)
print("Reporte",reporte)