#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 17:42:22 2024

@author: safat
"""

import cv2
import glob
import numpy as np

path = "/home/safat/Downloads/Optic-Disc-Segmentation-OpenCV-master/images/*"

for file in glob.glob(path):
    #print(file)
    a = cv2.imread(file)
    a = cv2.resize(a, (500,500))
    #print(a)
    #cv2.imshow('window - ' + file, a)
    file2 = file
    file2 = file2.replace("/home/safat/Downloads/Optic-Disc-Segmentation-OpenCV-master/images/", "")
    print(file2)
    
    
    cv2.imshow('window - ' + file2, a)
    
cv2.waitKey(0)
cv2.destroyAllWindows()