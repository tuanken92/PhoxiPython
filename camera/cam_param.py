class Cam_Param():
    def __init__(self, device_id:str, trigger_mode:str, cam_wd:float,  offset_width:int, offset_height:int) -> None:
        self.device_id = device_id
        self.cam_wd = cam_wd
        self.trigger_mode = trigger_mode
        self.offset_width = offset_width
        self.offset_height = offset_height

    def print_info(self):
        print("----------Camera Param-----------")
        print(f"device_id = {self.device_id}")
        print(f"cam_wd = {self.cam_wd}")
        print(f"trigger_mode = {self.trigger_mode}")
        print(f"offset_width = {self.offset_width}")
        print(f"offset_height = {self.offset_height}")
        print("----------Camera Param done-----------")