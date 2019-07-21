# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 15:23:23 2019

@author: MSI
"""

import cv2
import pandas as pd
import numpy as np
import os

#Just returns raw frames for manual editting and cropping
def simple_frame(vid):

    filename = "D:/query_data/facialrec/Videos/cinematics/" + vid + ".mp4"
    vidcap2 = cv2.VideoCapture(filename)
    success2,image2 = vidcap2.read()
    # I'm assuming directory was already made by me. Yeah, I'm nuts I know
    cin_dir = "D:/query_data/facialrec/Cinematic_Images/" + vid + "/"
    count2 = 0
    while success2:
        
        cv2.imwrite(cin_dir + vid + "_frame_" + str(count2) + ".jpg", image2)


        # cv2.imwrite("frame%d.jpg" % count2, image)     # save frame as JPEG file      
        success2,image2 = vidcap2.read()
        print('Read new frame: ', success2)
        count2 += 1