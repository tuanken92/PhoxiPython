import threading
from client.my_client import*
from lib.mylib import*
from camera.my_camera import*
from yl.my_detector import*

#variable
enable_thread_tcp = False
enable_thread_keyboard = False
device_id = ""
trigger = ""

#thread
thTCPClient = None
thKeyboard = None

#instance

client = None       #tcp client
camera = None       #camera
detector = None     #detector
config = None       #config


#fucntion

def run_thread():
    global client, camera, detector, config
    global thTCPClient, thKeyboard
    global enable_thread_tcp
    global enable_thread_keyboard
    global device_id



    thTCPClient = threading.Thread(target=process_cam_by_tcpip)
    thKeyboard = threading.Thread(target=process_keyboard)

    # Start the threads
    thTCPClient.start()
    thKeyboard.start()

    # Wait for both threads to finish
    thTCPClient.join()
    thKeyboard.join()

def init_proc():
    global client, camera, detector, config
    global thTCPClient, thKeyboard
    global enable_thread_tcp
    global enable_thread_keyboard
    global device_id, trigger


    t1 = current_milli_time()
    config_filename = 'configs/config.json' 
    config = read_config(config_filename)

    if config:
        server_ip = config.get("server_ip")
        server_port = config.get("server_port")
        api_key = config.get("api_key")

        debug_mode = config.get("debug_mode")

        device_id = config.get("device_id")
        trigger = config.get("trigger")
        
        cam_wd = config.get("cam_working_distance")

        model_path = config.get("model_path")

        print(f"Server IP: {server_ip}")
        print(f"Server Port: {server_port}")
        print(f"API Key: {api_key}")
        print(f"Debug Mode: {debug_mode}")
        print(f"device_id: {device_id}, WD = {cam_wd}, trigger = {trigger}")
        print(f"model_path: {model_path}")
    else:
        print("Configuration not loaded. Check your JSON file or path.")


    #client
    client = My_Client(server_ip, server_port)
    client.connect()

    #camera
    camera = My_Camera(device_id, cam_wd, trigger)
    camera.find_camera()
    is_connected = camera.connect()
    print("Is connected to camera {0} = {1}".format(device_id, is_connected))

    #detector
    detector = My_Detector(model_path)

    #enable thread
    enable_thread_keyboard = True
    enable_thread_tcp = True

    print("finish init program, took {0} ms".format(current_milli_time() - t1))

def process_cam_by_tcpip():
    global client, camera, detector, config
    global thTCPClient, thKeyboard
    global enable_thread_tcp
    global enable_thread_keyboard
    global device_id, trigger


    print(f"===========start thread tcp-client============")
    while enable_thread_tcp:
        if client == None or client.is_connected == False:
            time.sleep(0.05)
            continue
        
        #get data
        data_server = client.client_socket.recv(1024)
        if not data_server:
            print("=============Server closed the connection, close thread tcp-client")
            client.is_connected = False
            enable_thread_tcp = False
            break

        data = data_server.decode()
        print(f"data = {data}, type = {type(data)}")

        if data == None:
            time.sleep(0.05)
            continue

        #process data
        process_message(data)

        #clear data
        client.data = None
        
    print(f"===========finish thread tcp-client============")


def process_keyboard():
    global client, camera, detector, config
    global thTCPClient, thKeyboard
    global enable_thread_tcp
    global enable_thread_keyboard
    global device_id, trigger


    print(f"===========start thread keyboard============")
    while enable_thread_keyboard:
        message = input("Enter a message to send (or 'q' to quit): ")
        process_message(message)

    print(f"===========finish thread keyboard============")

def process_message(message):
    global client, camera, detector, config
    global thTCPClient, thKeyboard
    global enable_thread_tcp
    global enable_thread_keyboard
    global device_id, trigger


    print(f"===========data = {message}============")
    
    if message == 'q':
        enable_thread_tcp = False
        enable_thread_keyboard = False
        return

    if message == 'r':
        is_connected = camera.connect()
        print("Is connected to camera {0} = {1}".format(device_id, is_connected))

    elif message == 't':
        #get frame from camera
        frame = camera.trigger_camera()
        print("1---->frame data = ", frame.shape)
        #detect conner
        file_name_debug = "fr_current.bmp"
        b = cv2.imwrite(file_name_debug, frame)
        print("2----->connner begin, frame shape = {0}, save file {2}= {1}".format(frame.shape, b,file_name_debug))
        
        #detect box
        conners = detector.predict_frame(frame)
        box_data = camera.box_calculation(conners)
        if box_data == None:
            pass
        else:
            client.send_data(box_data)
        
    elif message == ' ':
        camera.trigger_camera_display()

    elif message == 'e':
        camera.close()

    elif message == 'load':
        detector.load_model()

    elif ".jpg" in message or ".png" in message or ".bmp" in message:
        detector.predict(message)

    print(f"==========done=============")

if __name__ == '__main__':
    print(f"==========staring program=============")
    #init program
    init_proc()

    #run thread
    run_thread()

    #close all connection
    camera.close()
    client.close()