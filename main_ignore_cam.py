import threading
from client.my_client import*
from lib.mylib import*
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
        model_path = config.get("model_path")

        print(f"Server IP: {server_ip}")
        print(f"Server Port: {server_port}")
        print(f"API Key: {api_key}")
        print(f"Debug Mode: {debug_mode}")
        print(f"device_id: {device_id}")
        print(f"model_path: {model_path}")
    else:
        print("Configuration not loaded. Check your JSON file or path.")


    #client
    client = My_Client(server_ip, server_port)
    client.connect()

    #detector
    detector = My_Detector(model_path)

    while True:
        message = input("Enter a message to send (or 'exit' to quit): ")
        if message.lower() == 'exit':
            break
        elif message.lower() == 'load':
            detector.load_model()
        elif ".jpg" in message.lower() or ".png" in message.lower() or ".bmp" in message.lower():
            detector.predict(message.lower())
            

        client.send_data(message)
    client.close()