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

def points_on_line_segment3(x1, y1, x2, y2):
    # Tính chiều dài của đoạn AB theo trục Ox và Oy
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    # Duyệt qua các điểm trên đoạn AB
    for i in range(max(dx, dy) + 1):
        x = x1 + i * (x2 - x1) // max(dx, dy)
        y = y1 + i * (y2 - y1) // max(dx, dy)
        yield x, y

def points_on_line_segment(p1, p2):
    data  = []
    # Tìm tọa độ tối thiểu và tối đa của đoạn thẳng
    Ax, Ay = p1
    Bx, By = p2
    
    Ax = int(Ax)
    Ay = int(Ay)
    Bx = int(Bx)
    By = int(By)

    min_x = min(Ax, Bx)
    max_x = max(Ax, Bx)

    # Tìm độ dài của đoạn thẳng trên trục Ox
    dx = max_x - min_x



    # Liệt kê các điểm nguyên trên đoạn thẳng AB
    if Ax <= Bx:
        # Tính hệ số góc của đoạn thẳng
        if dx != 0:
            m = (By - Ay) / dx
        else:
            m = 0

        for x in range(Ax, Bx+1):
            y = Ay + m * (x - Ax)
            rounded_y = round(y)  # Làm tròn tọa độ y để có giá trị nguyên
            # print(f"({x}, {rounded_y})")
            data.append([x,rounded_y])
    else:

        # Tính hệ số góc của đoạn thẳng
        if dx != 0:
            m = (Ay - By) / dx
        else:
            m = 0

        for x in range(Ax, Bx-1, -1):
            y = Ay + m * (x - Ax)
            rounded_y = round(y)  # Làm tròn tọa độ y để có giá trị nguyên
            # print(f"({x}, {rounded_y})")
            data.append([x,rounded_y])

    
    return data

def points_on_line_segment2(p1, p2):
    data  = []
    # Tính chiều dài của đoạn AB theo trục Ox và Oy
    x1, y1 = p1
    x2, y2 = p2

    print(f'x1 = {x1}; x2 = {x2}')
    print(f'y1 = {y1}; y2 = {y2}')

    dx = abs(int(x2) - int(x1))
    dy = abs(int(y2) - int(y1))

    
    print(f'dx = {dx}. dy = {dy}, max = {max(dx, dy)}')
    # return
    # Duyệt qua các điểm trên đoạn AB
    for i in range(max(dx, dy),1):
        x = x1 + i * (x2 - x1) // max(dx, dy)
        y = y1 + i * (y2 - y1) // max(dx, dy)
        print(f'x = {x}. y = {y}')
        data.append([x,y])
    
    return data

def process_line_point(conner_outside, conner_inside):
    result = []
    print(f'conner_outside = {conner_outside}')
    print(f'conner_inside = {conner_inside}')
    for i in range(4):
        print(f'i = {i}, conner_out = {conner_outside[i]}, conner_in = {conner_inside[i]}')
        point_data = points_on_line_segment(conner_outside[i], conner_inside[i])
        print(f'------------------{len(point_data)}')
        # print(f'point data = {point_data}')
        # for point in point_data:
        #     print(f'x = {point[0]}, y = {point[1]}')
        result.append(point_data)
        print(f'------------------')
    return result

def test():
    # Điểm A(x1, y1)
    x1 = 212
    y1 = 268

    # Điểm B(x2, y2)
    x2 = 223
    y2 = 289

    for x, y in points_on_line_segment(x1, y1, x2, y2):
        print(f"Điểm ({x}, {y}) trên đoạn thẳng AB trong hệ tọa độ Oxy")


def get_z_common(data_3d):
    if len(data_3d) == 0:
        return 0
    # Tạo một từ điển để đếm tần số xuất hiện của giá trị Z
    z_frequency = {}
    for point in data_3d:
        z_value = int(point[2])
        if z_value in z_frequency:
            z_frequency[z_value] += 1
        else:
            z_frequency[z_value] = 1

    # Tìm giá trị Z xuất hiện nhiều nhất
    max_z_value = max(z_frequency, key=z_frequency.get)
    max_frequency = z_frequency[max_z_value]

    print(f"Giá trị Z xuất hiện nhiều nhất: {max_z_value} (xuất hiện {max_frequency} lần)")
    return max_z_value
