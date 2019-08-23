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
    #cascPath = "C:/Users/MSI/Anaconda3/Library/etc/haarcascades/haarcascade_frontalface_default.xml"
    #cas = cv2.CascadeClassifier(cascPath)
    filename = "D:/query_data/facialrec/Videos/cinematics/" + vid
    vidcap2 = cv2.VideoCapture(filename)
    success2,image2 = vidcap2.read()
    # I'm assuming directory was already made by me. Yeah, I'm nuts I know
    # Disregard the last comment, I'm python god
    string = vid.replace(".mp4", "")
    cin_dir = "D:/query_data/facialrec/Cinematic_Images/" + string + "/"
    count2 = 0
    while success2:
        
        
        image2 = cv2.resize(image2, (200, 200), interpolation = cv2.INTER_AREA)
        cv2.imwrite(cin_dir + string + "_frame_" + str(count2) + ".jpg", image2)
        count2 += 1

        # cv2.imwrite("frame%d.jpg" % count2, image)     # save frame as JPEG file      
        success2,image2 = vidcap2.read()
        print('Read new frame: ', success2)
        count2 += 1
    