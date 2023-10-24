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
        self.Message = "NG"
        self.ImgURL = ""
        
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
        xxx =  "\"boxData\":{0}".format(json.dumps(self.to_dict(), indent=4))
        print(xxx)
        return xxx

    def box_NG(self):
        xxx =  "\"boxData\":{0}".format(json.dumps(self.to_dict(), indent=4))
        print(xxx)
        return xxx
