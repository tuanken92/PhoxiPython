import cv2
import numpy as np
import random
import os
import time
from collections import namedtuple, deque

import ultralytics
from ultralytics import YOLO
from ultralytics.utils import ops
import torch
print(torch.cuda.is_available())
import yaml

from PIL import Image
# from prepare_data import preprocess_image

# import pycoral
# from pycoral.adapters import common
# from pycoral.adapters.detect import get_objects
# from pycoral.utils.dataset import read_label_file
# from pycoral.utils.edgetpu import make_interpreter
# from pycoral.utils.edgetpu import run_inference


print("Ultralytics: ", ultralytics.__version__)
print("Torch: ", torch.__version__)
print("OpenCV: ", cv2.__version__)
# print("Pycoral: ", pycoral.__version__)

from scipy import special

DNNRESULT = namedtuple("DetResult", 
                       ["class_index", "box", "mask", "conf", "rect", "rect_offset", "rect_dim"], 
                       defaults=7*[None])

def plot_text(text, img:np.ndarray, org:tuple=None, color:tuple=None, line_thickness=5):
    """
    Helper function for drawing single min area rect on image
    Parameters:
        text : string
        img (np.ndarray): input image
    """
    color = color or (255, 255, 255)
    org = org or (10, 10)
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    tf = max(tl -1, 1)
    
    cv2.putText(img, text, org, 0, tl / 3, color, tf, cv2.LINE_AA)
    return img
    pass

def plot_one_min_rect(rect, img:np.ndarray, color:tuple=None, line_thickness=5):
    """
    Helper function for drawing single min area rect on image
    Parameters:
        rect :result of cv2.minAreaRect
        img (np.ndarray): input image
    """
    color = color or (255, 255, 255)
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    box = cv2.boxPoints(rect).astype(np.uintp)
    cv2.drawContours(img, [box], 0, color, tl)
    return img
    pass

def plot_one_box(box:np.ndarray, img:np.ndarray, color=None, mask:np.ndarray = None, label:str = None, line_thickness:int = 5):
        """
        Helper function for drawing single bounding box on image
        Parameters:
            x (np.ndarray): bounding box coordinates in format [x1, y1, x2, y2]
            img (no.ndarray): input image
            color (Tuple[int, int, int], *optional*, None): color in BGR format for drawing box, if not specified will be selected randomly
            mask (np.ndarray, *optional*, None): instance segmentation mask polygon in format [N, 2], where N - number of points in contour, if not provided, only box will be drawn
            label (str, *optonal*, None): box label string, if not provided will not be provided as drowing result
            line_thickness (int, *optional*, 5): thickness for box drawing lines
        """
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        fs = tl / 1.0
        if color is None:
            color = [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=fs, thickness=tf)[0]

            if c1[1] > t_size[1] + 3:
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                org = (c1[0], c1[1] -2)
            else:
                c2 = c1[0] + t_size[0], c1[1] + t_size[1] + 3
                org = (c1[0], c1[1] + t_size[1] + 2)

            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, org, 0, fs, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        if mask is not None:
            image_with_mask = img.copy()
            cv2.fillPoly(image_with_mask, pts=[mask], color=color)
            img = cv2.addWeighted(img, 0.5, image_with_mask, 0.5, 1)
        return img

def plot_results(results, img:np.ndarray, label_map={}, colors=[]):
        """
        Helper function for drawing bounding boxes on image
        Parameters:
            results: list DNNRESULT("class_index", "box", "mask", "conf")
            source_image (np.ndarray): input image for drawing
            label_map; (Dict[int, str]): label_id to class name mapping
        Returns:

        """
        result:DNNRESULT = None
        for result in results:
            box = result.box
            mask = result.mask
            rect = result.rect
            rect_offset = result.rect_offset
            rect_dim = result.rect_dim
            cls_index = result.class_index
            conf = result.conf

            h, w = img.shape[:2]

            if label_map:
                label = f'{label_map[cls_index]}, dim({rect_dim[0]:.2f};{rect_dim[1]:.2f};{rect_dim[2]:.2f}), score: {conf:.2f}'
            else:
                label = f"OBJ, score: {conf:.2f}"
            if len(colors):
                color = colors[cls_index]
            else:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            
            
            if mask is not None:
                print(f'rect = {rect}')
                print(f'rect_offset = {rect_offset}')
                color = (0, 255, 255)
                plot_one_min_rect(rect, img, 
                                color=color, 
                                line_thickness=2)
                color = (255, 0, 255)
                plot_one_min_rect(rect_offset, img, 
                                color=color, 
                                line_thickness=2)

            img = plot_one_box(box, img, 
                                mask=mask, 
                                label=label, 
                                color=color, line_thickness=1)
        return img