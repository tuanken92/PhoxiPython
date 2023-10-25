import time
import json
import math
from datetime import datetime


class Detector_Param():
    def __init__(self, model, img_size, conf, saved, offset_width, offset_height) -> None:
        self.model_path = model
        self.imgsz = img_size
        self.conf = conf
        self.saved_img = saved
        self.offset_width = offset_width
        self.offset_height = offset_height

    def print_info(self):
        print("----------Detector Param-----------")
        print(f"model_path = {self.model_path}")
        print(f"img_size = {self.imgsz}")
        print(f"conf = {self.conf}")
        print(f"saved image = {self.saved_img}")
        print(f"offset_width = {self.offset_width}")
        print(f"offset_height = {self.offset_height}")
        print("----------Detector Param done-----------")

        

def current_milli_time():
    return round(time.time() * 1000)

def print_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

# Define a function to read the configuration from the JSON file
def read_config(filename):
    try:
        with open(filename, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{filename}' not found.")
        return {}

def p2p(point1, point2):
    if len(point1) != 3 or len(point2) != 3:
        raise ValueError("Both points must be 3D coordinates with x, y, and z values.")

    distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2)
    return distance

def get_high_average(point1, point2, point3, point4, cam_setup):
    if len(point1) != 3 or len(point2) != 3 or len(point3) != 3 or len(point4) != 3:
        raise ValueError("Both points must be 3D coordinates with x, y, and z values.")

    h = cam_setup - (point1[2] + point2[2] + point3[2] + point4[2])/4
    return h
