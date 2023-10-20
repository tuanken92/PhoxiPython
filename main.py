import threading
from client.my_client import*
from lib.mylib import*
from camera.my_camera import*
from yl.my_detector import*

if __name__ == '__main__':
    config_filename = 'configs/config.json' 
    config = read_config(config_filename)

    if config:
        server_ip = config.get("server_ip")
        server_port = config.get("server_port")
        api_key = config.get("api_key")
        debug_mode = config.get("debug_mode")
        device_id = config.get("device_id")
        cam_wd = config.get("cam_working_distance")

        model_path = config.get("model_path")

        print(f"Server IP: {server_ip}")
        print(f"Server Port: {server_port}")
        print(f"API Key: {api_key}")
        print(f"Debug Mode: {debug_mode}")
        print(f"device_id: {device_id}, WD = {cam_wd}")
        print(f"model_path: {model_path}")
    else:
        print("Configuration not loaded. Check your JSON file or path.")


    #client
    client = My_Client(server_ip, server_port)
    client.connect()

    #camera
    camera = My_Camera(device_id, cam_wd)
    camera.find_camera()
    is_connected = camera.connect()
    print("Is connected to camera {0} = {1}".format(device_id, is_connected))

    #detector
    detector = My_Detector(model_path)

    while True:
        message = input("Enter a message to send (or 'q' to quit): ")
        if message.lower() == 'q':
            break
        elif message.lower() == 't':
            #get frame from camera
            frame = camera.trigger_camera()
            print("1---->frame data = ", frame.shape)
            #detect conner
            file_name_debug = "huhu.bmp"
            b = cv2.imwrite(file_name_debug, frame)
            print("2----->connner begin, frame shape = {0}, save file {2}= {1}".format(frame.shape, b,file_name_debug))
            conners = detector.predict_frame(frame)
            camera.box_calculation(conners)

            # camera.getPointCloud(451,118)
            # for point in conners:
            #     x, y = point
            #     print(f"Point: ({x}, {y})")
            #     camera.getPointCloud(y,x)
            #get point cloud from conner
            print("2----->connner",conners)
        elif message.lower() == ' ':
            camera.trigger_camera_display()
        elif message.lower() == 'e':
            camera.close()
        elif message.lower() == 'load':
            detector.load_model()
        elif ".jpg" in message.lower() or ".png" in message.lower() or ".bmp" in message.lower():
            detector.predict(message.lower())
            

        client.send_data(message)
    camera.close()
    client.close()