# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:19:55 2019

@author: MSI
"""

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os
import json
from urllib.request import *
import sys
import time
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary


def get_art(name, direc):
    download_path = direc
    searchtext = name + " lol"
    num_requested = 5000
    number_of_scrolls = 13 
	# number_of_scrolls * 400 images will be opened in the browser

    if not os.path.exists(download_path + searchtext.replace(" ", "_")):
        os.makedirs(download_path + searchtext.replace(" ", "_"))

    url = "https://www.google.co.in/search?q="+searchtext+"&source=lnms&tbm=isch"
    driver = webdriver.Firefox(executable_path = 'D:/query_data/facialrec/geckodriver.exe')
    driver.get(url)

    headers = {}
    headers['User-Agent'] = "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36"
    extensions = {"jpg", "jpeg", "png", "gif"}
    img_count = 0
    downloaded_img_count = 0
	
    for _ in range(int(number_of_scrolls)):
        for __ in range(10):
			# multiple scrolls needed to show all 400 images
            driver.execute_script("window.scrollBy(0, 1000000)")
            time.sleep(0.2)
		# to load next 400 images
        time.sleep(0.5)
        try:
            driver.find_element_by_xpath("//input[@value='Show more results']").click()
        except Exception as e:
            print ("Less images found: {}".format(e))
            break

	# imges = driver.find_elements_by_xpath('//div[@class="rg_meta"]') # not working anymore
    imges = driver.find_elements_by_xpath('//div[contains(@class,"rg_meta")]')
    print ("Total images: {}\n".format(len(imges)))
    for img in imges:
        img_count += 1
        img_url = json.loads(img.get_attribute('innerHTML'))["ou"]
        img_type = json.loads(img.get_attribute('innerHTML'))["ity"]
        print ("Downloading image {}:{}".format(img_count,img_url))
        try:
            if img_type not in extensions:
                img_type = "jpg"
            req = Request(img_url, headers=headers)
            raw_img = urlopen(req).read()
            f = open(download_path+searchtext.replace(" ", "_") + "_"+str(downloaded_img_count)+".jpg", "wb")
            f.write(raw_img)
            f.close
            downloaded_img_count += 1
        except Exception as e:
            print ("Download failed: {}".format(e))
        finally:
            print
        if downloaded_img_count >= num_requested:
            break

    print ("Total downloaded: {}/{}".format(downloaded_img_count,img_count))
    driver.quit()

