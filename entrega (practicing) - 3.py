#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 04:19:40 2024

@author: safat
"""

import numpy as np
#import scipy as sp 
import scipy.ndimage as ndimage
#import math
#from skimage import exposure,filters
from matplotlib import pyplot as plt
import cv2
#import matplotlib.pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks,hough_ellipse
#from skimage.feature import canny
from skimage.draw import circle_perimeter
#from skimage.segmentation import active_contour
#from skimage.filters import gaussian
#import argparse
from sklearn.cluster import KMeans

def resize(img):  
	return cv2.resize(img, (1080, 720))

def rgb2Blue(img):
	b,g,r = cv2.split(img)
	return b

def rgb2Red(img):
	b,g,r = cv2.split(img)
	return r

def rgb2Green(img):
	b,g,r = cv2.split(img)
	return g

def rgb2Gray(img):
	return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def rgb2lab(img):
	return cv2.cvtColor(img,cv2.COLOR_BGR2LAB)

#############preprocess###########
##Image is split on B,G,R channel
##Red channel is isolated
##Smoothing over the red channel is applied
##Sharpening and Equalization to te image are applied
##A morph closing is applied to remove artifacts
##################################
def preprocess(img):
	b,g,r = cv2.split(img)
	gray = rgb2Red(img)
	gray_blur = cv2.GaussianBlur(gray, (5,5), 0)
	gray = cv2.addWeighted(gray, 1.5, gray_blur, -0.5, 0, gray)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31))
	gray = ndimage.grey_closing(gray,structure=kernel)
	gray = cv2.equalizeHist(gray)	
	return gray

#############getROI##############
##Image is resized
##We take green channel and smooth it
##Opening is done to remove artifacts, in order to preserve only BRIGHTEST elements
##Now we get the most bright pixel position
##We return that position in a 110x110 window
##It is actually a simple way to detect the optic disc, but it works so..
##################################
def getROI(image):
	image_resized = resize(image)
	b,g,r = cv2.split(image_resized)
	g = cv2.GaussianBlur(g,(15,15),0)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
	g = ndimage.grey_opening(g,structure=kernel)	
	(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(g)

	x0 = int(maxLoc[0])-110
	y0 = int(maxLoc[1])-110
	x1 = int(maxLoc[0])+110
	y1 = int(maxLoc[1])+110
	
	return image_resized[y0:y1,x0:x1]


def getValue(img):
	shapeRow = img.shape[0]
	shapeCol = img.shape[1]
	x = 0 
	y = 0
	acu = 0
	maxloc = []
	for i in range(shapeRow):
		for j in range(shapeCol):
			(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img[i-15:j-15,i+15:j+15])
			value = maxVal
			if value > acu:
				acu = value 
				maxloc = maxLoc
	return maxloc

def kmeans(img):
	## K-Means
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	flags = cv2.KMEANS_RANDOM_CENTERS
	roi = img
	X = roi.reshape((-1, 1))
	X = np.float32(X)
	compactness,labels,centers = cv2.kmeans(X,3,None,criteria,10,flags)

	result = np.choose(labels, centers)
	result.shape = X.shape

	centers = np.uint8(centers)
	res= centers[labels.flatten()]
	res2 = res.reshape((roi.shape))
	return res2


def checkSide(img):
	shapeRow = img.shape[0]
	shapeCol = img.shape[1]
	if cv2.countNonZero(img[:,0:int(shapeCol/2)]) > cv2.countNonZero(img[:,int(shapeCol/2):shapeCol]):
		return True
	else:
		return False

def checkHigh(img):
	shapeRow = img.shape[0]
	shapeCol = img.shape[1]
	if cv2.countNonZero(img[0:int(shapeRow/2),:]) > cv2.countNonZero(img[int(shapeRow/2):shapeRow,:]):
		return True
	else:
		return False

def canny(img,sigma):
	v = np.mean(img)
	sigma = sigma
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(img, lower, upper)	
	return edged

def hough(edged,limm,limM):
	hough_radii = np.arange(limm, limM, 1)
	hough_res = hough_circle(edged, hough_radii)
	return hough_circle_peaks(hough_res, hough_radii,total_num_peaks=1)

'''
parser = argparse.ArgumentParser(description='Optic disc segmentation.')
parser.add_argument("-f", dest='file', action='store', type=str, help='The image to process.')
args = parser.parse_args()

print(args.accumulate(args.integers))
'''
############################
##Here we start the process
############################

image = cv2.imread("images/healthy1.jpg")
#image2 = cv2.imread("r2_Im320.png")
image = cv2.resize(image, (1080, 720))
cv2.imshow("original image", image)
title1 = 'original image'
figure1 = image


roi = getROI(image)
preprocessed_roi = preprocess(roi)
title2 = 'roi'
figure2 = roi

cv2.imshow("safat", roi)
#cv2.imshow("safat", preprocessed_roi)

complete_image = roi

#im2, contours, hierarchy = cv2.findContours(segmented, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
##We get our ROI and then we preprocess it.
##After preprocess, we apply a canny border algorithm
##We dilate the borders in order to make it easy to detect with hough


edged = canny(preprocessed_roi,0.22)
kernel = np.ones((3,3),np.uint8)

edged = cv2.dilate(edged,kernel,iterations=3)
accums, cx, cy, radii = hough(edged,55,80)



#########################################
#########################################
#########################################

image_hist = cv2.equalizeHist(preprocessed_roi)
cv2.imshow("equalized histogram", image_hist)
#print(image_hist.shape)
title3 = 'equalized histogram'
figure3 = image_hist

binary_img = (image_hist <= 240)

height, width = binary_img.shape

binary_img_2 = np.zeros((height,width))

data_points = []

for i in range(height):
    for j in range(width):
        if(binary_img[i][j] == True):
            binary_img_2[i][j] = 255
            
        else:
            binary_img_2[i][j] = 0
            data_points.append((i,j))


cv2.imwrite("binary image.jpg", binary_img_2)

img2 = cv2.imread("binary image.jpg")
cv2.imshow("binary image", img2)
title4 = 'Binary Threshold image'
figure4 = img2

print(img2.shape)

#########################################
#########################################
#########################################


# Create KMeans instance and fit the data
kmeans = KMeans(n_clusters=1)
kmeans.fit(data_points)

centroids = kmeans.cluster_centers_

cup_x = centroids[0][0]
cup_y = centroids[0][1]

cup_x = np.uint8(cup_x)
cup_y = np.uint8(cup_y)

print(cup_x, cup_y)

img2[cup_x, cup_y] = (255, 255, 0)
complete_image[cup_y, cup_x] = (255, 255, 0)
#cv2.imshow('image with centroid', img2)
cup_radius = 0
total_dist = 0


for i in range(len(centroids)):
    point_x, point_y = np.int8(data_points[i])
    
    dist = np.sqrt((point_x - cup_x)**2 + (point_y - cup_y)**2)    
    total_dist += dist
    
cup_radius = total_dist/len(centroids)
cup_radius = np.uint8(cup_radius)

img2 = cv2.circle(img2, (cup_y, cup_x), cup_radius, (255,255,0))
complete_image = cv2.circle(complete_image, (cup_x, cup_y), cup_radius, (255,255,0))
cv2.imshow('cup radius with circle', img2)
title5 = 'cup radius with circle'
figure5 = img2
#########################################
#########################################
#########################################

center_x = 0
center_y = 0
radius = 0
for center_y, center_x, radius in zip(cy, cx, radii):
	circy, circx = circle_perimeter(center_y, center_x, radius)
	try:
		roi[circy, circx] = (220, 20, 20)
	except :
		continue


#init = np.array([circx, circy]).T


cv2.imshow('Pre-Processed Region of Interest ', preprocessed_roi)
cv2.imshow('Region of Interest ', roi)
title6 = 'Pre-Processed Region of Interest'
figure6 = preprocessed_roi

img2[center_x, center_y] = (0,0,255)
img2 = cv2.circle(img2, (center_y, center_x), radius, (0,0,255))

complete_image[center_y, center_x] = (255,0,0)
#complete_image = cv2.circle(complete_image, (center_x, center_y), radius, (255,0,0))

cv2.imshow('updated img2', img2)
title7 = 'After all pre processing'
figure7 = img2

ellipsis_height = 5
axesLength=(cup_radius-ellipsis_height, cup_radius+ellipsis_height)
center_coordinates=(cup_y, cup_x)
angle=0
startAngle=0 
endAngle=360 
color=(255,255,0) 
thickness=3
img2 = cv2.ellipse(img2, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness)

cv2.imshow('img2', img2)
############################ display all images #################

titles = [title1, title2, title3, title4, title5, title6, title7]
figures = [figure1, figure2, figure3, figure4, figure5, figure6, figure7]
'''
for i in range(7):
    plt.subplot(3, 3, i+1), plt.imshow(figures[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
    
plt.show()
'''

cv2.imshow('complete image', complete_image)

cv2.waitKey(0)
cv2.destroyAllWindows()