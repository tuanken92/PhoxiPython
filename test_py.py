import cv2
import numpy as np

import operator
from stat import ST_CTIME
import os, sys, time
my_dir = 'frame/' 
import glob
import os
def get_last_file():
    list_of_files = []
    # os.chdir(my_dir)
    for file in glob.glob("frame/*.png"):
        # print(file)
        list_of_files.append(file)
        
    # list_of_files = glob.glob(f'./frame/*.png') # * means all if need specific format then *.csv
    if len(list_of_files) == 0:
        return None

    latest_file = max(list_of_files, key=os.path.getctime)
    # xxx =  f'{my_dir}{latest_file}'
    # print(f'call latest_file = {latest_file}')
    return latest_file

init_file = "result_inference.png"
last_file = 'frame/abc.png'
name_wd = "3D_data"
current_file = ""

cv2.namedWindow(name_wd, cv2.WINDOW_AUTOSIZE)
x = 10
y = 212
w = 726
h = 362

while True:
    last_file = get_last_file()
    if last_file == None:
        print(f'last_file = None')
        last_file = init_file
    # else:
    #     if last_file != init_file:
    #         last_file = init_file
    
    
    if current_file != last_file:
        print(f'last file = {last_file}, init_file = {init_file}')
        current_file = last_file
        img = cv2.imread(current_file)
        
        crop_img = img[y:y+h, x:x+w]
        # print(f'img shape = {img.shape}')
        cv2.imshow(name_wd, crop_img)
        cv2.waitKey(1)
    
    time.sleep(0.5)