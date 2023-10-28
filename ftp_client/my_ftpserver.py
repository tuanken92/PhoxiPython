import ftplib
import cv2

from lib.mylib import*
from pathlib import Path
from ftp_client.ftp_param import*

# session = ftplib.FTP('127.0.0.1','thh','gachick@10')
# file = open('detector_hehe.png','rb')                  # file to send
# session.storbinary('STOR /robot/detector.jpg', file)     # send the file
# file.close()                                    # close file and FTP
# session.quit()



class My_FTPUpload:
    def __init__(self, ftp_param:FTP_Param) -> None:
        self.session = ftplib.FTP(ftp_param.ftp_server,ftp_param.user,ftp_param.password)
        self.ftp_parm = ftp_param
        self.file_on_server = None

    def upload_file(self, file_path:str):
        if True:
            print(f'[DATA] ====> upload file: file path = {file_path}')
            #open file
            file = open(file_path,'rb')
            
            # get file name
            file_name = Path(file_path).stem
            # print(f'====> upload file: file name = {file_name}')
            
            self.file_on_server = f'{self.ftp_parm.dir_upload}/{file_name}.png'
            # send the file
            self.session.storbinary(f'STOR {self.file_on_server}', file)

            #close file
            file.close()

            # print(f'====> upload file: file ftp = {self.file_on_server}')
            return self.file_on_server

        # except:
        #     print(f"can't send file {self.file_on_server}")
        #     return None


    def close_fpt(self):
        self.session.quit()
