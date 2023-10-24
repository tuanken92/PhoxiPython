import ftplib
from pathlib import Path


# session = ftplib.FTP('127.0.0.1','thh','gachick@10')
# file = open('detector_hehe.png','rb')                  # file to send
# session.storbinary('STOR /robot/detector.jpg', file)     # send the file
# file.close()                                    # close file and FTP
# session.quit()



class My_FTPUpload:
    def __init__(self, fpt, user, port) -> None:
        self.session = ftplib.FTP(fpt,user,port)
        self.file_on_server = None

    def upload_file(self, file_path):
        if True:
            print(f'====> upload file: file path = {file_path}')
            #open file
            file = open(file_path,'rb')
            
            # get file name
            file_name = Path(file_path).stem
            print(f'====> upload file: file name = {file_name}')
            
            self.file_on_server = f'logistic/{file_name}.png'
            # send the file
            self.session.storbinary(f'STOR {self.file_on_server}', file)

            #close file
            file.close()

            print(f'====> upload file: file ftp = {self.file_on_server}')
            return self.file_on_server

        # except:
        #     print(f"can't send file {self.file_on_server}")
        #     return None

    def close_fpt(self):
        self.session.quit()
