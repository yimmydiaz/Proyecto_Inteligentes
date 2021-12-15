import numpy as np
import cv2
from prediccion import Prediccion
import base64
import requests
import json

nameWindow = "Calculadora"
img_counter = 0
listVertices = []
#clases = ["cepillo", "llaves", "maquina","tijera"]
listaImagenes = []
listaModelos = []

ancho = 256
alto = 256

def nothing(x):
	pass

def constructorVentana():
	cv2.namedWindow(nameWindow)
	cv2.createTrackbar("min",nameWindow, 0, 255, nothing)
	cv2.createTrackbar("max",nameWindow, 100, 255, nothing)
	cv2.createTrackbar("kernel",nameWindow, 1, 100, nothing)
	cv2.createTrackbar("areaMin",nameWindow, 500, 10000, nothing)

def calcularAreas(figuras):
	areas = []
	for figuraActual in figuras:
		areas.append(cv2.contourArea(figuraActual))
	return areas

img_counter = 0

def recortar(figura):
	global img_counter
	img_name = "imagen.png"
	x, y, w, h = cv2.boundingRect(figura)
	new_img=imagen[y:y+h,x:x+w]
	new_img = cv2.resize(new_img, (1889, 1889))
	cv2.imwrite(img_name,new_img)
	image = open('imagen.png', 'rb')
	image_read = image.read()
	image_64_encode = base64.encodebytes(image_read)
	print("Imagen Agregada")
	listaImagenes.append(image_64_encode.decode())



def enviarPeticion():
	print("Llamado al Backend")
	objecto = {'imagen': listaImagenes, 'models': listaModelos}
	resp = requests.post('http://127.0.0.1:5000/predict', json=json.dumps(objecto), stream=True)
	print(resp.json())

def detectarForma(imagen):
	global img_counter
	imagenGris = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
	min = cv2.getTrackbarPos("min", nameWindow)
	max = cv2.getTrackbarPos("max", nameWindow)
	bordes = cv2.Canny(imagenGris, min, max)
	tamañoKernel = cv2.getTrackbarPos("kernel", nameWindow)
	kernel = np.ones((tamañoKernel,tamañoKernel), np.uint8)
	bordes = cv2.dilate(bordes, kernel)
	cv2.imshow("Bordes Reforzados", bordes)
	figuras, jerarquia = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	areas = calcularAreas(figuras)
	areaMin = cv2.getTrackbarPos("areaMin", nameWindow)
	i = 0
	for figuraActual in figuras:
		if areas[i] >= areaMin:
			#mensaje = "Cuadrado"+str(areas[i])
			#cv2.putText(imagen, mensaje, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.drawContours(imagen, [figuraActual], 0, (0,0,255), 2)
			if cv2.waitKey(1)%256 == 99: #c:
				recortar(figuraActual)
				#resultadoPredi = resultadoPrediccion()
			if cv2.waitKey(1)%256 == 101: #e:
				enviarPeticion()
		i = i + 1
	return imagen


camara = cv2.VideoCapture(1)
constructorVentana()

while(True) :
	_,imagen = camara.read()
	imagen = detectarForma(imagen)
	cv2.imshow('Imagen Camara Cliente', imagen)  #Mostrar imagen
	k = cv2.waitKey(5) & 0xFF #ESC para salir
	if k == 27:
		break

camara.release()
cv2.destroyAllWindows()













"""def resultadoPrediccion():
	miModeloCNN = Prediccion("models/modeloA.h5", ancho, alto)
	imagen = cv2.imread("imagen.png")
	claseResultado = miModeloCNN.predecir(imagen)
	print("el valor es", clases[claseResultado])
	
	def verificarVertices(vertices):
	verificacion = True
	x, y, w, h = cv2.boundingRect(vertices)
	for item in listVertices:
		xAux, yAux, wAux, hAux = cv2.boundingRect(item)
		if(x==xAux and y==yAux and w==wAux and h==hAux):
			verificacion = False
			return verificacion
	return verificacion
	
	"""