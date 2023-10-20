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

class My_Camera:
    def __init__(self, device_id, cam_wd) -> None:
        self.point_cloud_component = None
        self.point_cloud = None
        self.cam_width = 0

        self.device_id = device_id
        self.cam_working_distance = cam_wd
        self.cam = None
        self.cam_feature = None
        self.is_connected = False
        
        #thread
        self.process_thread = None

        #harvester to find camera
        self.h = Harvester()
        if platform == "win32":
            cti_file_path_suffix = "/API/bin/photoneo.cti"
        else:
            cti_file_path_suffix = "/API/lib/photoneo.cti"
        cti_file_path = os.getenv('PHOXI_CONTROL_PATH') + cti_file_path_suffix
        print("--> cti_file_path: ", cti_file_path)
        self.h.add_file(cti_file_path, True, True)
        

    def find_camera(self):
        self.h.update()
        # Print out available devices
        print()
        print("Name : ID")
        print("---------")
        for item in self.h.device_info_list:
            print(item.property_dict['serial_number'], ' : ', item.property_dict['id_'])
        print()

    def configure_camera(self):
        #print(dir(features))
        print("TriggerMode BEFORE: ", self.features.PhotoneoTriggerMode.value)
        self.features.PhotoneoTriggerMode.value = "Software"
        print("TriggerMode AFTER: ", self.features.PhotoneoTriggerMode.value)

        # Order is fixed on the selected output structure. Disabled fields are shown as empty components.
        # Individual structures can enabled/disabled by the following features:
        # SendTexture, SendPointCloud, SendNormalMap, SendDepthMap, SendConfidenceMap, SendEventMap, SendColorCameraImage
        # payload.components[#]
        # [0] Texture
        # [1] TextureRGB
        # [2] PointCloud [X,Y,Z,...]
        # [3] NormalMap [X,Y,Z,...]
        # [4] DepthMap
        # [5] ConfidenceMap
        # [6] EventMap
        # [7] ColorCameraImage

        # Send every output structure
        self.features.SendTexture.value = True
        self.features.SendPointCloud.value = True
        self.features.SendNormalMap.value = False
        self.features.SendDepthMap.value = False
        self.features.SendConfidenceMap.value = False

        #start camera
        self.cam.start()
    
    def trigger_camera(self):
        # Trigger frame by calling property's setter.
        # Must call TriggerFrame before every fetch.
        self.features.TriggerFrame.execute() # trigger first frame

        # also possible use with error checking:
        self.features.TriggerFrame.execute() # trigger third frame
        # buffer = self.cam.try_fetch(timeout=10.0) # grab newest frame
        # if buffer is None:
        #     # check if device is still connected
        #     self.is_connected = self.features.IsConnected.value
        #     if not self.is_connected:
        #         print('Device disconnected!--------------reconnect camera please!')
        # # ...
        # # work with buffer
        # # ...
        # payload = buffer.payload
        # texture_rgb_component = payload.components[1]
        # frame = get_texture(texture_rgb_component, "TextureRGB")
        # # release used buffer (only limited buffers available)
        # buffer.queue()
        # return frame
        with self.cam.fetch(timeout=10.0) as buffer:
            # grab first frame
            # do something with first frame
            print("buffer = ",buffer)

            # The buffer object will automatically call its dto once it goes
            # out of scope and releases internal buffer object.
            payload = buffer.payload
            self.point_cloud_component = payload.components[2]
            self.cam_width = self.point_cloud_component.width
            self.point_cloud = self.point_cloud_component.data.reshape(self.point_cloud_component.height * self.point_cloud_component.width, 3).copy()
            self.getPointCloud(451,118)

            #text ture 
            #texture_component = payload.components[0]
            #display_texture_if_available(texture_component)
            
            texture_rgb_component = payload.components[1]
            return get_texture(texture_rgb_component, "TextureRGB")
            
    
    def trigger_camera_display(self):
        # Trigger frame by calling property's setter.
        # Must call TriggerFrame before every fetch.
        self.features.TriggerFrame.execute() # trigger first frame
        with self.cam.fetch(timeout=10.0) as buffer:
            # grab first frame
            # do something with first frame
            print(buffer)

            # The buffer object will automatically call its dto once it goes
            # out of scope and releases internal buffer object.
            payload = buffer.payload
            self.point_cloud_component = payload.components[2].copy()
            self.getPointCloud(451,118)


            texture_component = payload.components[0]
            display_texture_if_available(texture_component)
            
            texture_rgb_component = payload.components[1]
            display_color_image_if_available(texture_rgb_component, "TextureRGB")

            color_image_component = payload.components[7]
            display_color_image_if_available(color_image_component, "ColorCameraImage")

            #3d visualize
            # point_cloud_component = payload.components[2]
            # norm_component = payload.components[3]
            # display_pointcloud_if_available(point_cloud_component, norm_component, texture_component, texture_rgb_component)

    def connect(self):
        try:
            #connect cam with ID
            self.cam = self.h.create({'id_': self.device_id})
            self.features = self.cam.remote_device.node_map
            self.is_connected = True

            #config camera
            self.configure_camera()
            
            #run thread
            self.is_connected = True
            # self.receive_thread = threading.Thread(target=self.receive_data_thread)
            # self.receive_thread.daemon = True
            # self.receive_thread.start()

        except:
            self.is_connected = False
            print(f"exception: Connect to camera error, check connection....!")
        return self.is_connected

    def box_calculation(self, conners):
        print(conners)
        if conners.size == 0:
            print(f"Not found conner, can't get box dim, try again!")
            return


        p = []
        for index, item in enumerate(conners):
            print(index, item)
            x, y = item
            print(f"Point: ({x}, {y})")
            p.append(self.getPointCloud(y,x))
        
        #get distance from p2p
        w = p2p(p[0], p[1])
        h = p2p(p[1], p[2])
        z = get_high_average(p[0], p[1], p[2], p[3], self.cam_working_distance)
        print(f"box dim = {w},{h},{z}")

    def getPointCloud(self, y, x):
        #convert point cloud
        
        print("pointcloud shape = {0}".format(self.point_cloud.shape))

        
        print("cam_width = {0}".format(self.cam_width))

        #get index
        index = (int)(y * self.cam_width + x)
        print(f"point cloud index = {index}")
        #get data
        print("data 2d ({0},{1}) => 3d ({2}), index = {3}".format(x,y,self.point_cloud[index],index))
        return self.point_cloud[index]


    def close(self):
        if self.is_connected:
            self.is_connected = False
            # if self.receive_thread:
            #     self.receive_thread.join()
            self.cam.stop()
            self.cam.destroy()
        
