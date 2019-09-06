# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import cv2
import numpy as np
import os
from get_art import get_art
from cin_images import cin_images
import os.path
#setup

git_dir = "C:/Users/MSI/Documents/Github/FacialRecAnalysis/"
vid_dir = "D:/query_data/facialrec/Videos/"
img_dir = "D:/query_data/facialrec/Images/"
val_dir = "D:/query_data/facialrec/Validation/"
#cascPath = "C:/Users/MSI/Anaconda3/Library/etc/haarcascades/haarcascade_frontalface_default.xml"
#cas = cv2.CascadeClassifier(cascPath)

champs = ["Camille", "Diana", "Ezreal", "Graves", "Jayce", "Karma", "Lucian", "Syndra", "Talon", "Vi"]

coords = pd.read_csv(git_dir + "Data/coords.csv")

#for each in champs
for i in range(len(champs)):
    name = champs[i]
    
    #making training data directory
    os.makedirs(img_dir + name, exist_ok = True)
    #validation set directory
    os.makedirs(val_dir + name, exist_ok = True)
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
        print('Read a new ' + name + ' frame: ', success)
        count += 1
        
    
    '''
    #models
    num = os.listdir(vid_dir + "models/" + name) # dir is your directory path
    nfiles = len(num)
    count2 = 0
    model_dir = champ_dir + "/models/"
    os.makedirs(model_dir, exist_ok = True)
    for j in range(nfiles):
        
        vidcap2 = cv2.VideoCapture(vid_dir + "models/" + name + "/video_" + str(j) + ".mp4")
        success2,image2 = vidcap2.read()
        
        while success2:
            
            image2 = cv2.resize(image2, (200, 200), interpolation = cv2.INTER_AREA)
            cv2.imwrite(model_dir + name + "_model_frame_" + str(count2) + ".jpg", image2)
            success2,image2 = vidcap2.read()
            print('Read new ' + name + ' frame: ', success2)
            count2 += 1
    '''
    '''
    #getting art data
    art_dir = val_dir + name
    os.makedirs(art_dir, exist_ok = True)
    get_art(name, art_dir + "/")
    
    #cinematic images
    
    champ_cin_dir = "D:/query_data/facialrec/Images/" + name + "/cinematics/"
    os.makedirs(champ_cin_dir, exist_ok = True)
    
#screw it, I'm not* hard coding because I'm a madman
cin_img_dir = "D:/query_data/facialrec/Cinematic_Images/"
cin_vid_dir = vid_dir + "cinematics"
number = os.listdir(vid_dir + "cinematics")
nvids = len(number)
for k in range(nvids):
    filename = os.listdir(vid_dir + "cinematics/")[k]
    cin_images(filename)
'''