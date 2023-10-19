import socket
import threading



class My_Client:
    def __init__(self, ip_server, port) -> None:
        self.data = None
        self.server_ip = ip_server
        self.server_port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.receive_thread = None
        self.is_connected = False

    def connect(self):
        try:
            self.client_socket.connect((self.server_ip, self.server_port))
            self.is_connected = True
            print(f"Connected to {self.server_ip}:{self.server_port}")
            self.receive_thread = threading.Thread(target=self.receive_data_thread)
            self.receive_thread.daemon = True
            self.receive_thread.start()
        except ConnectionRefusedError:
            self.is_connected = False
            print(f"Connection to {self.server_ip}:{self.server_port} refused")
            return

    def send_data(self, data):
        if self.is_connected:
            self.client_socket.send(data.encode())
        else:
            print("Check connection...")

    def receive_data_thread(self):
        while self.is_connected:
            data = self.client_socket.recv(1024)
            if not data:
                print("Server closed the connection.")
                self.is_connected = False
                break
            message = data.decode()
            self.on_data_received(message)  # You can implement your event handling logic here

    def on_data_received(self, data):
        # This is where you can implement your event handling logic
        print(f"Received: {data}")


    def close(self):
        self.is_connected = False
        if self.receive_thread:
            self.receive_thread.join()
        self.client_socket.close()
