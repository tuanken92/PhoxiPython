import ftplib

def tuanna_callback(abc):
    print(f"========done===========")


session = ftplib.FTP('127.0.0.1','thh','gachick@10')
file = open('detector_hehe.png','rb')                  # file to send
session.storbinary('STOR /robot/detector.jpg', file, callback=tuanna_callback)     # send the file
file.close()                                    # close file and FTP
session.quit()