#!/usr/bin/env python

#Std Libs
import cv2

#Ros Python Libs
import rospy
import ros_numpy as rnp
from cv_bridge import CvBridge

#ROS msgs
from sensor_msgs.msg import Image

#Image Processing
from ultralytics import YOLO

class PersonLocator():
	def __init__(self):
		self.camera_node_name = "/usb_cam/image_raw"
		self.model = YOLO('yolov8n.pt')
		rospy.loginfo('YOLOv8 is now running...')
		self.sub = rospy.Subscriber(self.camera_node_name, Image, self.callback)
		self.yolo_pub = rospy.Publisher('yolo_pub', Image, queue_size=10)
		self.bridge = CvBridge()  # Para convertir entre ROS y OpenCV
		self._image_data = None

	def callback(self, msg):
		img = rnp.numpify(msg)  # Convertir el mensaje de imagen a formato NumPy
		results = self.model(img, conf=0.65)  # Ejecutar el modelo YOLO en la imagen

		# Convertir la imagen NumPy a formato OpenCV
		img_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

		# Filtrar solo las detecciones de la clase "person" (etiqueta 0 en YOLO)
		for result in results:
			for detection in result.boxes:
				if detection.cls == 0:  # Clase '0' es "person"
					# Dibujar las detecciones de personas
					x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Coordenadas de la caja
					cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dibuja el rectángulo
					label = f"Person: {detection.conf.item():.2f}, size x = {x2-x1} size y = {y2-y1}"  # Etiqueta con la confianza
					cv2.putText(img_cv2, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# Convertir la imagen procesada de nuevo a un mensaje ROS
		img_msg = self.bridge.cv2_to_imgmsg(img_cv2, encoding="bgr8")
		self.yolo_pub.publish(img_msg)  # Publicar la imagen con las detecciones de personas

if __name__=='__main__':
	rospy.init_node('person_locator')
	person_locator = PersonLocator()
	rospy.spin()

