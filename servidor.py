from flask import Flask, render_template, jsonify
from flask.globals import request
import base64
from keras.preprocessing.image import ImageDataGenerator
from scipy import ndimage
import numpy as np
import imageio
import sys
import json
from prediccion import Prediccion
import cv2
from os import remove
from os import path
import time

app = Flask(__name__)
clases = [ "bisturi","cepillo","llaves","maquina","tijera"]

@app.route("/")
def home():
    return "<h1>sever up</h1>"


@app.route("/predict", methods=["POST", "GET"])
def predict():

    if request.method == "POST":
        data = json.loads(request.get_json(force=True))
        images = data["imagen"]
        models = data["models"]
        response = procesarImagen(images)
        if response["status"] == 200:
            response["info"]["state"] = "success"
            response["info"]["message"] = "Predictions made satisfactorily"
        elif response["status"] == 400:
            response["info"] = {}
            response["info"]["state"] = "error"
            response["info"]["message"] = "Error making predictions"

        return app.make_response((jsonify(response["info"]), response["status"]))

    elif request.method == "GET":
        user = "request.get_json()"
        return user

    else:
        return render_template()


def procesarImagen(imagenes):
    ancho = 256
    alto = 256
    i = 1
    resultObjects = []
    miModeloCNNA = Prediccion("models/modeloA.h5", ancho, alto)
    miModeloCNNB = Prediccion("models/modeloB.h5", ancho, alto)
    miModeloCNNC = Prediccion("models/modeloC.h5", ancho, alto)
    miModeloCNND = Prediccion("models/modeloD.h5", ancho, alto)
    miModeloCNNE = Prediccion("models/modeloE.h5", ancho, alto)
    try:
        for image in imagenes:
            imgdata = base64.b64decode(image)
            if path.exists("imagen/captura.png"):
                remove('imagen/captura.png')
            filename = 'imagen/captura.png'
            with open(filename, "wb") as f:
                f.write(imgdata)
            imagen = cv2.imread(filename)
            #modeloA
            inicio_tiempo = time.time()
            claseResultado = miModeloCNNA.predecir(imagen)
            final_tiempo = time.time() - inicio_tiempo
            result = clases[claseResultado]
            objecto = {'model_A':1, 'imagen': i, 'result': result}
            print("Tiempo de Espera", i, final_tiempo)
            resultObjects.append(objecto)

             #modeloB
            inicio_tiempo = time.time()
            claseResultado = miModeloCNNB.predecir(imagen)
            final_tiempo = time.time() - inicio_tiempo
            result = clases[claseResultado]
            objecto = {'model_B':2, 'imagen': i, 'result': result}
            print("Tiempo de Espera", i, final_tiempo)
            resultObjects.append(objecto)

            #modeloC
            inicio_tiempo = time.time()
            claseResultado = miModeloCNNC.predecir(imagen)
            final_tiempo = time.time() - inicio_tiempo
            result = clases[claseResultado]
            objecto = {'model_C':3, 'imagen': i, 'result': result}
            print("Tiempo de Espera", i, final_tiempo)
            resultObjects.append(objecto)

            #modeloD
            inicio_tiempo = time.time()
            claseResultado = miModeloCNND.predecir(imagen)
            final_tiempo = time.time() - inicio_tiempo
            result = clases[claseResultado]
            objecto = {'model_D': 4, 'imagen': i, 'result': result}
            print("Tiempo de Espera", i, final_tiempo)
            resultObjects.append(objecto)

            #modeloE
            inicio_tiempo = time.time()
            claseResultado = miModeloCNNE.predecir(imagen)
            final_tiempo = time.time() - inicio_tiempo
            result = clases[claseResultado]
            objecto = {'model_E': 5, 'imagen': i, 'result': result}
            print("Tiempo de Espera", i, final_tiempo)
            resultObjects.append(objecto)

            i = i + 1
        return  {"info": {"response": resultObjects}, "status": 200}
    except Exception as e:
        print(e)
        return {"info":{},"status":400}

if __name__ == "__main__":
    app.run(debug=True)