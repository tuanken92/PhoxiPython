class FTP_Param():
    def __init__(self, ftp_server:str, user:str, password:str, dir_upload='logistics') -> None:
        self.ftp_server = ftp_server
        self.user = user
        self.password = password
        self.dir_upload = dir_upload


    def print_info(self):
        print("----------FTP Param-----------")
        print(f"ftp_server = {self.ftp_server}")
        print(f"user = {self.user}")
        print(f"password = {self.password}")
        print(f"dir_upload = {self.dir_upload}")
        print("----------FTP Param done-----------")