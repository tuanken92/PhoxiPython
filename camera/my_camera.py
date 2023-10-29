import socket
import threading
import numpy as np

#import open3d as o3d
import cv2
import os
import sys
import my_param

from sys import platform
from harvesters.core import Harvester
from camera.camera_func import*
from box.box import*
from vision_dl.drawing import*
from camera.cam_param import*
from ftp_client.my_ftpserver import*

class My_Camera:
    def __init__(self, camera_param:Cam_Param, ftp_client:My_FTPUpload) -> None:
        self.point_cloud_component = None
        self.point_cloud = None
        self.cam_width = 0

        #ftp client
        self.ftp_client = ftp_client

        self.cam_param = camera_param
        self.cam = None
        self.cam_feature = None
        self.is_connected = False
        self.is_found_cam = False
        self.my_frame = None
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
        self.is_found_cam = False
        # Print out available devices
        print()
        print("Name : ID")
        print("---------")
        for item in self.h.device_info_list:
            print(item.property_dict['serial_number'], ' : ', item.property_dict['id_'])
            if self.cam_param.device_id == item.property_dict['id_']:
                self.is_found_cam = True
        print()



    def configure_camera(self):
        #print(dir(features))
        print("TriggerMode BEFORE: ", self.features.PhotoneoTriggerMode.value)
        self.features.PhotoneoTriggerMode.value = self.cam_param.trigger_mode
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
        # self.features.TriggerFrame.execute() # trigger first frame
        # with self.cam.fetch(timeout=10.0) as buffer:
                # grab first frame
                # do something with first frame
                # pass
        # also possible use with error checking:
        self.features.TriggerFrame.execute() # trigger third frame
        with self.cam.fetch(timeout=10.0) as buffer:
            # grab first frame
            # do something with first frame
            # print("buffer = ",buffer)

            # The buffer object will automatically call its dto once it goes
            # out of scope and releases internal buffer object.
            payload = buffer.payload
            self.point_cloud_component = payload.components[2]
            self.cam_width = self.point_cloud_component.width
            self.point_cloud = self.point_cloud_component.data.reshape(self.point_cloud_component.height * self.point_cloud_component.width, 3).copy()
            #self.getPointCloud(451,118)

            #text ture 
            #texture_component = payload.components[0]
            #display_texture_if_available(texture_component)
            
            texture_rgb_component = payload.components[1]
            self.my_frame = get_texture(texture_rgb_component, "TextureRGB")
            return self.my_frame
            

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
        if self.is_found_cam:
            #connect cam with ID
            self.cam = self.h.create({'id_': self.cam_param.device_id})
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

    def box_calculation(self, conners, ftp_link):
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
        box.Message = "OK"
        box.ImgURL = ftp_link
        return box.to_json()
        
    def box_ng(self):
        #upload current frame
        ftp_local_path = "logo.jpg"
        img_url = self.ftp_client.upload_file(ftp_local_path)
        #format box ng
        box = BOX()
        box.Name = "BOX"
        box.Message = "NG"
        box.ImgURL = img_url
        data = (box.to_json())
        return data

    def box_ok(self, result:DNNRESULT, label_map:dict):
        print(f'\tfinnal_result = {result}')
        t1 = current_milli_time()
        nc = len(label_map)
        colors = np.random.uniform(0, 255, size=(nc, 3))
        mat = plot_results([result], self.my_frame, label_map=label_map, colors=colors)
        mat = self.my_frame
        # mat = cv2.resize(mat, (640,480))
        #upload current frame
        file_name = f'frame/frame_{current_milli_time()}.png'
        is_saved_file = cv2.imwrite(file_name, mat)
        if not is_saved_file:
            print("==========> save file NG===========>")
            return self.box_ng()

        img_url = self.ftp_client.upload_file(file_name)

        #format box ng
        box = BOX()
        box.Name = label_map[result.class_index]
        box.Message = "OK"
        box.width = result.rect_dim[0]      #w
        box.length = result.rect_dim[1]     #h
        box.height = result.rect_dim[2]     #z
        box.ImgURL = img_url
        data = (box.to_json())
        t2 = current_milli_time() -t1
        print("============== total box OK = {0} ms=================".format(current_milli_time() -t1))
        return data

    def box_calculation2(self, results, label_map:dict):
        """
        results: list DNNRESULT("class_index", "box", "mask", "conf",...)
        """
        if len(results) == 0:
            return self.box_ng()

        result:DNNRESULT = None

        #filter bbox in frame
        t1 = current_milli_time()
        results_in_frame = []
        for result in results:
            rect_offset = result.rect_offset
            box_with_padding = cv2.boxPoints(rect_offset).astype(np.uintp)
            is_ok = True
            for conner in box_with_padding:
                x,y = conner
                if x >= self.point_cloud_component.width or y>=self.point_cloud_component.height:
                    is_ok = False
                    # print(f"Ignore index {index}, data = {box_with_padding}")
                    break
            if is_ok:
                # print(f"Add index {index}, data = {box_with_padding}")
                results_in_frame.append(result)
        #print(results_in_frame)
        t2 = current_milli_time()
        # print("============== filter1 = {0} ms=================".format(current_milli_time() -t1))
        if len(results_in_frame) == 0:
            print("============== No Box, send box NG and return=================")
            return self.box_ng()

        #filter bbox with max confidence in frame
        result_max_conf:DNNRESULT = None
        max_confident = 0.0
        for result in results_in_frame:
            conf = result.conf
            if conf >= max_confident:
                max_confident = conf
                result_max_conf = result
                # print(f"===========Update conf max = {conf}")
        t3 = current_milli_time()
        # print("============== filter2 = {0} ms=================".format(current_milli_time() -t2))
        #print(result_max_conf)
        if result_max_conf == None:
            print("============== No Box, send box NG and return=================")
            return self.box_ng()

        #get box dimension from conner
        rect_offset = result_max_conf.rect_offset
        conners = cv2.boxPoints(rect_offset).astype(np.uintp)
        p =[]
        for conner in conners:
            x,y=conner
            p.append(self.getPointCloud(y,x))
        
        #get distance from p2p
        w = p2p(p[0], p[1])
        h = p2p(p[1], p[2])
        z = get_high_average(p[0], p[1], p[2], p[3], self.cam_param.cam_wd)
        box_dim = [w, h, z]
        finnal_result = result_max_conf._replace(rect_dim=box_dim)
        # print(f'\tfinnal_result = {finnal_result}')
        # print("============== get box = {0}=================".format(current_milli_time() -t3))
        t4 = current_milli_time() - t1
        # print(f"============== BOX_CALCULATOR {t2}ms=================")
        return self.box_ok(finnal_result,label_map)

    def getPointCloud(self, y:int, x:int):
        #convert point cloud
        # print("pointcloud shape = {0}".format(self.point_cloud.shape))
        # print("cam_width = {0}".format(self.cam_width))
        #get index
        index = (int)(y * self.cam_width + x)
        # print(f"point cloud index = {index}")
        #get data
        print("\tMapping 2d ({0},{1}) => 3d ({2}), index = {3}".format(x,y,self.point_cloud[index],index))
        return self.point_cloud[index]


    def close(self):
        if self.is_connected:
            self.is_connected = False
            # if self.receive_thread:
            #     self.receive_thread.join()
            # Remove all callbacks to not any callback work:
            # self.cam.remove_callbacks()
            # self.cam.remove_callback(self.cam.Events.NEW_BUFFER_AVAILABLE)
            # self.cam.d()
            # self.cam.destroy()
        
