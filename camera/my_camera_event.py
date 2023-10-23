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
from box.box import*

from harvesters.core import Harvester
from harvesters.core import ImageAcquirer
from typing import Optional
from harvesters.core import Callback


class _OnNewBufferAvailable(Callback):
    def __init__(self, ia: ImageAcquirer, buffers: list):
        super().__init__()
        print("====================> _OnNewBufferAvailable init")
        self._ia = ia
        self._buffers = buffers

    def emit(self, context: Optional[object] = None) -> None:
        print("====================> _OnNewBufferAvailable emit")
        buffer = self._ia.fetch(timeout=10.0)
        print("====================> _OnNewBufferAvailable emit1")
        # self._buffers.append(buffer)
        buffer.queue()
        print("====================> _OnNewBufferAvailable emit2")
        #isssue: it can't break here, infinitive here forever

    @property
    def buffers(self):
        return self._buffers


class _OnReturnBufferNow(Callback):
    def __init__(self, holder: _OnNewBufferAvailable):
        super().__init__()
        print("====================> _OnReturnBufferNow init")
        self._holder = holder
        
    def emit(self, context: Optional[object] = None) -> None:
        # Return/Queue the buffers before stopping image acquisition:
        print("====================> _OnReturnBufferNow emit1")
        while len(self._holder.buffers) > 0:
            print("====================> _OnReturnBufferNow emit2")
            buffer = self._holder.buffers.pop(-1)
            buffer.queue()
            print("====================> _OnReturnBufferNow emit3")

class My_Camera:
    def __init__(self, device_id, cam_wd, trig) -> None:
        self.point_cloud_component = None
        self.point_cloud = None
        self.cam_width = 0
        self._buffers = []

        self.device_id = device_id
        self.trigger = trig
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
        self.features.PhotoneoTriggerMode.value = self.trigger
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


        # Create a callback:
        self.on_new_buffer_available = _OnNewBufferAvailable(
            ia=self.cam, buffers=self._buffers
        )
        self.on_return_buffer_now = _OnReturnBufferNow(
            holder=self.on_new_buffer_available
        )

        # event happened:
        self.cam.add_callback(
            ImageAcquirer.Events.NEW_BUFFER_AVAILABLE,
            self.on_new_buffer_available
        )
        self.cam.add_callback(
            ImageAcquirer.Events.RETURN_ALL_BORROWED_BUFFERS,
            self.on_return_buffer_now
        )

        #start camera
        self.cam.start(run_as_thread=True)
    
    def trigger_camera(self):
        # Trigger frame by calling property's setter.
        # Must call TriggerFrame before every fetch.
        self.features.TriggerFrame.execute() # trigger first frame
        with self.cam.fetch(timeout=10.0) as buffer:
                # grab first frame
                # do something with first frame
                pass
        # also possible use with error checking:
        self.features.TriggerFrame.execute() # trigger third frame
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
            

    def trigger_camera2(self):
        # Trigger frame by calling property's setter.
        # Must call TriggerFrame before every fetch.
        self.features.TriggerFrame.execute() # trigger first frame
        with self.cam.fetch(timeout=10.0) as buffer:
                # grab first frame
                # do something with first frame
                pass
        # also possible use with error checking:
        self.features.TriggerFrame.execute() # trigger third frame
       
        with self.cam.fetch(timeout=10.0) as buffer:
            # grab first frame
            # do something with first frame
            print("buffer = ",buffer)

            # The buffer object will automatically call its dto once it goes
            # out of scope and releases internal buffer object.
            payload = buffer.payload
            # self.point_cloud_component = payload.components[2]
            # self.cam_width = self.point_cloud_component.width
            # self.point_cloud = self.point_cloud_component.data.reshape(self.point_cloud_component.height * self.point_cloud_component.width, 3).copy()
            # self.getPointCloud(451,118)


            # Create a NumPy array from the image data
            texture_rgb_component = payload.components[1]

            height = texture_rgb_component.height
            width =  texture_rgb_component.width
            data = texture_rgb_component.data.copy()
            # color_image = color_component.data.reshape(color_component.height, color_component.width, 3).copy()
            data_array = np.frombuffer(data, dtype=np.uint16)

            print(f"h={height}, w = {width}, size = {height*width}, data size = {data_array.size}")
            # Reshape the data to match the image dimensions
            matYCoCg = np.reshape(data_array, (height, width,3))
            matRgb = YCoCg.convertToRGB(matYCoCg)

            # # Convert YCoCg to RGB
            # rgb_mat = YCoCg.convertToRGB(mat)
            # print("======> rgb_mat shape = {0}".format(rgb_mat.shape))

            # # Normalize for visualization
            # rgb_mat = cv2.normalize(matRgb, None, 0, 255, cv2.NORM_MINMAX)
            # rgb_mat = rgb_mat.astype(np.uint8)

            # cv2.imshow uses BGR format, so convert back to BGR
            #matRgb = cv2.cvtColor(matRgb, cv2.COLOR_RGB2BGR)
            # print("======> rgb_mat shape2 = {0}".format(rgb_mat.shape))
            
            return matRgb

    def cvt_color(self, color_component, name):
        if color_component.width == 0 or color_component.height == 0:
            print(name + " is empty!")
            return None
        
        # Reshape 1D array to 2D RGB image
        color_image = color_component.data.reshape(color_component.height, color_component.width, 3).copy()
        # Normalize array to range 0 - 65535
        #color_image = cv2.normalize(color_image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        color_image = color_image.astype(np.int16)
        #color_image = color_image.astype(np.int8)
        # Show image
        #cv2.imshow(name, color_image)
        b = cv2.imwrite("fr_{0}.bmp".format(current_milli_time()), color_image)
        print("save frame = {0}".format(b))
        return color_image

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
        if True:
            #connect cam with ID
            self.cam = self.h.create({'id_': self.device_id})
            print(self.cam.Events.__members__)
            
            self.features = self.cam.remote_device.node_map
            self.is_connected = True

            #config camera
            self.configure_camera()
            
            #run thread
            # self.receive_thread = threading.Thread(target=self.receive_data_thread)
            # self.receive_thread.daemon = True
            # self.receive_thread.start()

        else:
            self.is_connected = False
            print(f"exception: Connect to camera error, check connection....!")
        return self.is_connected

    def box_calculation(self, conners):
        print("conners", conners)
        print("conners type", type(conners))
        if len(conners) == 0:
            print(f"Not found conner, can't get box dim, try again!")
            return None


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
        box = BOX()
        box.height = z
        box.width = h
        box.length = w
        return box.to_json()

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
            # Remove all callbacks to not any callback work:
            self.cam.remove_callbacks()
            # self.cam.remove_callback(self.cam.Events.NEW_BUFFER_AVAILABLE)
            self.cam.stop()
            # self.cam.destroy()
        
