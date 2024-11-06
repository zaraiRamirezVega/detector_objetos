import cv2

thres = 0.5

# Ajustar el tamaño de la ventana de la cámara a una resolución mayor (por ejemplo, 1280x720)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Ancho de la ventana
cap.set(4, 720)   # Alto de la ventana

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # Usar un cuadro rojo para los objetos detectados
            cv2.rectangle(img, box, color=(0, 0, 255), thickness=2)  # Rojo

            # Cambiar el color del texto a blanco (o amarillo) para mejor visibilidad
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)  # Blanco
            # También cambiar el color del texto de la confianza a blanco (o amarillo)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)  # Blanco

    # Mostrar la imagen con las detecciones
    cv2.imshow("Output", img)
    cv2.waitKey(1)


coco.names
frozen_inference_graph.pb
main.py
ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt