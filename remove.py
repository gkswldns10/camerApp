# This is to remove certain parts of a picture in python

import numpy as np
import cv2

import matplotlib.pyplot as plt

# pre-trained face detector & classifier in opencv
cv_path = "/usr/local/lib/python3.5/dist-packages/cv2/data"
class_path = cv_path + "/haarcascade_frontalface_default.xml" # Classifier path

class Classifier():
	def __init__(self):
		# Obtain haarcascade_frontalface_default classifier in my computer
		self.face_cascade = cv2.CascadeClassifier(class_path)

	# Return grayscale of image (reduce to 1 channel)
	def GrayScale(self,img):
		gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		return gray_scale

	# Assume there is a face in the picture
	def DetectFaces(self,img):
		gray_scale = self.GrayScale(img)
		faces = self.face_cascade.detectMultiScale(image = gray_scale, scaleFactor = 1.2, minNeighbors = 5)
		if(len(faces) == 0):
			return None
		return faces

	# using canny algorithm
	# To find where the edges are at, use np.where(edges > 0)
	def DetectEdges(self,img):
		edges = cv2.Canny(img, 100, 200) # threshold1, threshold2
		return edges

	# Crop all faces from img
	def CropAll(self,img):
		faces = self.DetectFaces(img)
		gray_scale = self.GrayScale(img)
		for (x,y,w,h) in faces:
			cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)
			roi_gray = gray_scale[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]
		return roi_color,(np.arange(y,y+h), np.arange(x,x+w))

# Read a file in cv2 format (uint8)
def CVRead(file):
	return cv2.imread(file)


# plt.figure()
# edges = DetectEdges(img)
# plt.imshow(edges, cmap = 'gray')
# plt.show()

# img = CVRead("me.jpg")
# al,indices = CropAll(img)
# cv2.imshow('img',al)
# cv2.waitKey(0)
# cv2.destroyAllWindows()