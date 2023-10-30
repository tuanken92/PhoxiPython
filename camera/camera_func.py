import socket
import threading
import numpy as np
#import open3d as o3d
import cv2
import os
import sys
from sys import platform
from harvesters.core import Harvester
from lib.mylib import*



def display_texture_if_available(texture_component):
    if texture_component.width == 0 or texture_component.height == 0:
        print("Texture is empty!")
        return
    
    # Reshape 1D array to 2D array with image size
    texture = texture_component.data.reshape(texture_component.height, texture_component.width, 1).copy()
    rows, cols, channels = texture.shape
    print("texture shape = {0}".format(texture.shape))
    texture_screen = cv2.normalize(texture, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    # Show image
    cv2.imshow("Texture", texture_screen)
    return

def display_color_image_if_available(color_component, name):
    if color_component.width == 0 or color_component.height == 0:
        print(name + " is empty!")
        return
    
    # Reshape 1D array to 2D RGB image
    color_image = color_component.data.reshape(color_component.height, color_component.width, 3).copy()
    # Normalize array to range 0 - 65535
    color_image = cv2.normalize(color_image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)

    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    # Show image
    cv2.imshow(name, color_image)
    
    cv2.imwrite("xxx_{0}.bmp".format(current_milli_time()), color_image)
    return color_image

def get_texture(color_component, name):
    if color_component.width == 0 or color_component.height == 0:
        print(name + " is empty!")
        return None
    
    # Reshape 1D array to 2D RGB image
    color_image = color_component.data.reshape(color_component.height, color_component.width, 3).copy()
    # Normalize array to range 0 - 65535
    # color_image = cv2.normalize(color_image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    color_image = color_image.astype(np.int16)
    # color_image = color_image.astype(np.int16)
    # Show image
    #cv2.imshow(name, color_image)
    return color_image
    my_frame = "frame/my_frame.bmp"
    b = cv2.imwrite(my_frame, color_image)
    print("\tSave frame {0} = {1}".format(my_frame, b))
    return cv2.imread(my_frame)




def display_pointcloud_if_available(pointcloud_comp, normal_comp, texture_comp, texture_rgb_comp):
    # if pointcloud_comp.width == 0 or pointcloud_comp.height == 0:
    #     print("PointCloud is empty!")
    #     return
    
    # # Reshape for Open3D visualization to N x 3 arrays
    # pointcloud = pointcloud_comp.data.reshape(pointcloud_comp.height * pointcloud_comp.width, 3).copy()
    # print("pointcloud shape = {0}".format(pointcloud.shape))

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pointcloud)

    # if normal_comp.width > 0 and normal_comp.height > 0:
    #     norm_map = normal_comp.data.reshape(normal_comp.height * normal_comp.width, 3).copy()
    #     pcd.normals = o3d.utility.Vector3dVector(norm_map)

    # # Reshape 1D array to 2D (3 channel) array with image size
    # texture_rgb = np.zeros((pointcloud_comp.height * pointcloud_comp.width, 3))
    # if texture_comp.width > 0 and texture_comp.height > 0:
    #     texture = texture_comp.data.reshape(texture_comp.height, texture_comp.width, 1).copy()
    #     texture_rgb[:, 0] = np.reshape(1/65536 * texture, -1)
    #     texture_rgb[:, 1] = np.reshape(1/65536 * texture, -1)
    #     texture_rgb[:, 2] = np.reshape(1/65536 * texture, -1)        
    # elif texture_rgb_comp.width > 0 and texture_rgb_comp.height > 0:
    #     texture = texture_rgb_comp.data.reshape(texture_rgb_comp.height, texture_rgb_comp.width, 3).copy()
    #     texture_rgb[:, 0] = np.reshape(1/65536 * texture[:, :, 0], -1)
    #     texture_rgb[:, 1] = np.reshape(1/65536 * texture[:, :, 1], -1)
    #     texture_rgb[:, 2] = np.reshape(1/65536 * texture[:, :, 2], -1)
    # else:
    #     print("Texture and TextureRGB are empty!")
    #     return
    # texture_rgb = cv2.normalize(texture_rgb, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # pcd.colors = o3d.utility.Vector3dVector(texture_rgb)
    # o3d.visualization.draw_geometries([pcd], width=800,height=600)
    return
