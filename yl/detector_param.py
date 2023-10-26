class Detector_Param():
    def __init__(self, model_path:str, label_path:str, img_size:int, conf:float,
                        saved:bool, offset_width:int, offset_height:int) -> None:
        self.model_path = model_path
        self.label_path = label_path
        self.imgsz = img_size
        self.conf = conf
        self.saved_img = saved
        self.offset_width = offset_width
        self.offset_height = offset_height

    def print_info(self):
        print("----------Detector Param-----------")
        print(f"model_path = {self.model_path}")
        print(f"label_path = {self.label_path}")
        print(f"img_size = {self.imgsz}")
        print(f"conf = {self.conf}")
        print(f"saved image = {self.saved_img}")
        print(f"offset_width = {self.offset_width}")
        print(f"offset_height = {self.offset_height}")
        print("----------Detector Param done-----------")

