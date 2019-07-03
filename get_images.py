# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import cv2
import numpy as np
import os

#setup
git_dir = "C:/Users/MSI/Documents/Github/FacialRecAnalysis/"
vid_dir = "D:/query_data/facialrec/Videos/"
img_dir = "D:/query_data/facialrec/Images/"
cascPath = "C:/Users/MSI/Anaconda3/Library/etc/haarcascades/haarcascade_frontalface_default.xml"
cas = cv2.CascadeClassifier(cascPath)

champs = ["Camille", "Diana", "Ezreal", "Graves", "Jayce", "Karma", "Lucian", "Syndra", "Talon", "Vi"]

coords = pd.read_csv(git_dir + "Data/coords.csv")

#for each in champs
for i in range(len(champs)):
    name = champs[i]
    os.makedirs(img_dir + name, exist_ok = True)
    champ_dir  = img_dir + name
    #logins
    vidcap = cv2.VideoCapture(vid_dir + "logins/" + name + '/video_login.mp4')
    success,image = vidcap.read()
    count = 0
    login_dir = champ_dir + "/logins/"
    os.makedirs(login_dir, exist_ok = True)
    while success:
        x = coords['x'][i]
        y = coords['y'][i]
        w = coords['w'][i]
        h = coords['h'][i]                
        
        newimg = image[y:y+h, x:x+w]
        newimg = cv2.resize(newimg, (50, 50), interpolation = cv2.INTER_AREA)
        cv2.imwrite(login_dir + name + "_login_frame_" + str(count) + ".jpg", newimg)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
        
    
    vidcap2 = cv2.VideoCapture(vid_dir + "models/" + name + "/video.mp4")
    success2,image2 = vidcap2.read()
    count2 = 0
    model_dir = champ_dir + "/models/"
    os.makedirs(model_dir, exist_ok = True)
    while success2:
        
        gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
        faces = cas.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            flags = cv2.CASCADE_SCALE_IMAGE
            )

    # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            rect = image2[y:y+h, x:x+w]
            rect = cv2.resize(rect, (50, 50), interpolation = cv2.INTER_AREA)
            cv2.imwrite(model_dir + name + "_model_frame_" + str(count2) + ".jpg", rect)

        # cv2.imwrite("frame%d.jpg" % count2, image)     # save frame as JPEG file      
        success2,image2 = vidcap2.read()
        print('Read new frame: ', success2)
        count2 += 1
        
        