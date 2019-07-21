# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:29:16 2019

@author: MSI
"""

import cv2
import pandas as pd
import numpy as np
import os


def cin_images(vid):
    cascPath = "C:/Users/MSI/Anaconda3/Library/etc/haarcascades/haarcascade_frontalface_default.xml"
    cas = cv2.CascadeClassifier(cascPath)
    filename = "D:/query_data/facialrec/Videos/cinematics/" + vid + ".mp4"
    vidcap2 = cv2.VideoCapture(filename)
    success2,image2 = vidcap2.read()
    # I'm assuming directory was already made by me. Yeah, I'm nuts I know
    cin_dir = "D:/query_data/facialrec/Cinematic_Images/" + vid + "/"
    count2 = 0
    while success2:
        
        gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = cas.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50),
                flags = cv2.CASCADE_SCALE_IMAGE
                )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            rect = image2[y:y+h, x:x+w]
            rect = cv2.resize(rect, (50, 50), interpolation = cv2.INTER_AREA)
            cv2.imwrite(cin_dir + vid + "_frame_" + str(count2) + ".jpg", rect)
            count2 += 1

        # cv2.imwrite("frame%d.jpg" % count2, image)     # save frame as JPEG file      
        success2,image2 = vidcap2.read()
        print('Read new frame: ', success2)
        count2 += 1
    