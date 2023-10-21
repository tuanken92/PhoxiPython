import socket
import threading
import numpy as np
#import open3d as o3d
import cv2
import os
import sys

from sys import platform
from harvesters.core import Harvester
from camera.camera_func import*
from camera.YCoCg import*

class BOX:
    def __init__(self) -> None:
        self.width = 0.0
        self.height = 0.0
        self.length = 0.0

        self.Name = "box"
        self.Message = "OK"
        self.ImgURL = "ftp://192.168.100.111/logistics/box_dim.jpg"
        
    def to_dict(self):
        # Create a dictionary representing the object's attributes
        return {
            "width": self.width,
            "height": self.height,
            "length": self.length,
            "Name": self.Name,
            "Message": self.Message,
            "ImgURL": self.ImgURL
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)
