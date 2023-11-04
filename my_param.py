from lib.mylib import*
from yl.detector_param import*
from camera.cam_param import*
from ftp_client.ftp_param import*
from client.client_param import*


#instance
detector_param: Detector_Param = None
camera_param: Cam_Param = None
ftp_param: FTP_Param = None
client_param: Client_Param = None


def load_param_from_config():
    global detector_param
    global camera_param
    global ftp_param
    global client_param


    t1 = current_milli_time()
    config_filename = 'configs/config.json' 
    config = read_config(config_filename)

    if config:
        #server
        server_ip = config.get("server_ip")
        server_port = config.get("server_port")
        client_param = Client_Param(server_ip, server_port)
        client_param.print_info()

        #fpt server
        ftp_server = config.get("ftp_server")
        ftp_user = config.get("ftp_user")
        ftp_pass = config.get("ftp_pass")
        ftp_dir = config.get("ftp_dir")
        ftp_param = FTP_Param(ftp_server, ftp_user, ftp_pass, ftp_dir)
        ftp_param.print_info()

        #camera
        device_id = config.get("device_id")
        trigger_mode = config.get("trigger")
        cam_wd = config.get("cam_working_distance")
        offset_w = config.get("offset_w")
        offset_h = config.get("offset_h")
        camera_param = Cam_Param(device_id, trigger_mode, cam_wd, offset_w, offset_h)
        camera_param.print_info()

        #model DL
        model_path = config.get("model_path")
        label_path = config.get("label_path")
        score = config.get("score")
        saved = config.get("saved")
        img_size = config.get("img_size")
        detector_param = Detector_Param(model_path, label_path, img_size, score, 
                                        saved, offset_w, offset_h)

        detector_param.print_info()
        
    else:
        print("Configuration not loaded. Check your JSON file or path.")
    print("finish init program, took {0} ms".format(current_milli_time() - t1))
