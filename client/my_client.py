import socket
import threading

from client.client_param import*


class My_Client:
    def __init__(self, client_param:Client_Param) -> None:
        self.data = None
        self.client_param = client_param
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.receive_thread = None
        self.is_connected = False

    def connect(self):
        try:
            self.client_socket.connect((self.client_param.server_add, self.client_param.server_port))
            self.is_connected = True
            print(f"Connected to {self.client_param.server_add}:{self.client_param.server_port}")
            # self.receive_thread = threading.Thread(target=self.receive_data_thread)
            # self.receive_thread.daemon = True
            # self.receive_thread.start()
        except ConnectionRefusedError:
            self.is_connected = False
            print(f"Connection to {self.client_param.server_add}:{self.client_param.server_port} refused")
            return

    def send_data(self, data):
        if self.is_connected:
            self.client_socket.send(data.encode())
        else:
            print("Check connection...")

    def receive_data_thread(self):
        while self.is_connected:
            data_server = self.client_socket.recv(1024)
            if not data_server:
                print("Server closed the connection.")
                self.is_connected = False
                break
            self.data = data_server.decode()
            self.on_data_received(self.data)  # You can implement your event handling logic here

    def get_data_from_server(self):
        return self.data

    def on_data_received(self, data):
        # This is where you can implement your event handling logic
        print(f"Received: {data}")


    def close(self):
        self.is_connected = False
        if self.receive_thread:
            self.receive_thread.join()
        self.client_socket.close()
