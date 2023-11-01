import time
import json
import math
import yaml
from datetime import datetime


def load_labels(label_path:str):
    label_map = {}
    if label_path.endswith('.yaml'):
        with open(label_path, 'r', encoding="utf-8") as file:
            data = yaml.safe_load(file)
            label_map = data['names']

    elif label_path.endswith('.txt'):
        with open(label_path, 'r', encoding="utf-8") as file:
            lines = file.readlines()
            label_map = {}
            for i, line in enumerate(lines):
                line = line.strip("\r\n").split(" ")
                if len(line) > 1:
                    if line[0].isdigit():
                        label_map[int(line[0])] = " ".join(line[1:])
                    else:
                        label_map[i] = " ".join(line)    
                else:
                    label_map[i] = " ".join(line)
    return label_map



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

def points_on_line_segment(x1, y1, x2, y2):
    # Tính chiều dài của đoạn AB theo trục Ox và Oy
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    # Duyệt qua các điểm trên đoạn AB
    for i in range(max(dx, dy) + 1):
        x = x1 + i * (x2 - x1) // max(dx, dy)
        y = y1 + i * (y2 - y1) // max(dx, dy)
        yield x, y

def test():
    # Điểm A(x1, y1)
    x1 = 212
    y1 = 268

    # Điểm B(x2, y2)
    x2 = 223
    y2 = 289

    for x, y in points_on_line_segment(x1, y1, x2, y2):
        print(f"Điểm ({x}, {y}) trên đoạn thẳng AB trong hệ tọa độ Oxy")