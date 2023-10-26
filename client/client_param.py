class Client_Param():
    def __init__(self, server_add:str, server_port:str) -> None:
        self.server_add = server_add
        self.server_port = server_port

    def print_info(self):
        print("----------Client Param-----------")
        print(f"server_port = {self.server_port}")
        print(f"server_add = {self.server_add}")
        print("----------Client Param done-----------")