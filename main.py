import threading
from client.my_client import*
from lib.mylib import*
from camera.my_camera import*
from yl.my_detector import*
from ftp_client.my_ftpserver import*
from box.box import*
import my_param

#variable
enable_thread_tcp = False
enable_thread_keyboard = False

#thread
thTCPClient = None
thKeyboard = None

#instance
client:My_Client = None       #tcp client
camera:My_Camera = None       #camera
detector:My_Detector = None     #detector
ftp_client:My_FTPUpload = None   #ftp


#fucntion

def run_thread():
    global thTCPClient, thKeyboard

    thTCPClient = threading.Thread(target=process_cam_by_tcpip)
    thKeyboard = threading.Thread(target=process_keyboard)

    # Start the threads
    thTCPClient.start()
    thKeyboard.start()

    # Wait for both threads to finish
    thTCPClient.join()
    thKeyboard.join()

def init_proc():
    global client, camera, detector, ftp_client
    global enable_thread_tcp
    global enable_thread_keyboard

    #enable thread
    enable_thread_keyboard = True
    enable_thread_tcp = True

    #get current time
    t_start = current_milli_time()
    
    #load parameter from file
    my_param.load_param_from_config()

    #client
    client = My_Client(my_param.client_param)
    client.connect()
    
    #ftp
    ftp_client = My_FTPUpload(my_param.ftp_param)

    #camera
    camera = My_Camera(my_param.camera_param, ftp_client)
    camera.find_camera()
    is_connected = camera.connect()
    print("Is connected to camera {0} = {1}".format(my_param.camera_param.device_id, is_connected))

    
    #detector
    detector = My_Detector(my_param.detector_param)

    print("finish init program, took {0} ms".format(current_milli_time() - t_start))



def process_cam_by_tcpip():
    global client
    global enable_thread_tcp


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
    print(f"===========start thread keyboard============")
    while enable_thread_keyboard:
        message = input("Enter a message to send (or 'q' to quit): ")
        process_message(message)

    print(f"===========finish thread keyboard============")



def process_message(message):
    global client, camera, detector
    global thTCPClient, thKeyboard
    global enable_thread_tcp
    global enable_thread_keyboard


    print(f"===========data = {message}============")
    
    if message == 'q':
        enable_thread_tcp = False
        enable_thread_keyboard = False
        return

    if message == 'r':
        camera.find_camera()
        is_connected = camera.connect()
        print("Is connected to camera {0} = {1}".format(my_param.camera_param.device_id, is_connected))

    elif message == 't':
        #get frame from camera
        frame = camera.trigger_camera()

        # print("1---->frame data = ", frame.shape)
        # #detect conner
        file_name_debug = "fr_current.bmp"
        b = cv2.imwrite(file_name_debug, frame)
        print("----->save frame shape = {0}, save file {2}= {1}".format(frame.shape, b,file_name_debug))
        
        #detect box
        results = detector.predict_frame(frame)
        
        # ftp_file = ftp_client.upload_file(detector.saved_file_detector)
        # if not ftp_file:
            # print("=====> upload file error")

        #get box data
        # ftp_file = f"ftp://{my_param.ftp_param.ftp_server}/{ftp_file}"
        box_data = camera.box_calculation2(results, detector.label_map)
        if box_data == None:
            boxNG = BOX()
            # boxNG.ImgURL = ftp_file
            client.send_data(boxNG.box_NG())
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
    ftp_client.close_fpt()