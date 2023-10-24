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
        
    def upload_file(self, file_path):
        try:
            
            #open file
            file = open(file_path,'rb')
            
            # get file name
            file_name = Path(file_path).stem
            
            file_on_server = f'logistic/{file_name}.png'
            # send the file
            self.session.storbinary(f'STOR {file_on_server}', file)

            #close file
            file.close()

            print(f'====> upload file: file path = {file_path}')
            print(f'====> upload file: file name = {file_name}')
            print(f'====> upload file: file ftp = {file_on_server}')
            return file_on_server

        except:
            print(f"can't send file {file_on_server}")
            return None

    def close_fpt(self):
        self.session.quit()
