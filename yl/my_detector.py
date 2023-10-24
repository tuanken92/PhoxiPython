from ultralytics import YOLO
from ultralytics import settings
from lib.mylib import*
import numpy as np
from collections import namedtuple, deque
import cv2
import random

#print(settings)
DNNRESULT = namedtuple("DetResult", 
                       ["class_index", "box", "mask", "conf", "rect"], 
                       defaults=5*[None])

class My_Detector:
    def __init__(self, model_path) -> None:
        self.model_path = model_path
        self.loaded = False
        self.load_model()
        self.imgsz=640
        self.conf=0.65
        self.saved_file_detector = "detector.png"
        
    def load_model(self):
        try:
            # Load a model
            t1 = current_milli_time()
            print(f"Begin load model {self.model_path}...")
            self.model = YOLO(self.model_path)  # pretrained YOLOv8n model
            self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8))
            print("Loaded model, it tool {0} ms".format(current_milli_time() - t1))
            self.loaded = True
        except:
            print("can't load model!")
            self.loaded = False

    def predict2(self, source):
        if not self.loaded:
            print(f"Model not yet load, please load model and try again....")
            return None
        try:
            # Make predictions
            t1 = current_milli_time()
            print("start predict {0}".format(source))
            results = self.model.predict(source, save=False, imgsz=self.imgsz, conf=self.conf)
            print("--------->finnished, take {0} ms".format(current_milli_time() - t1))
            print(results[0])
            return results
        except:
            print(f"Can't predict")
        return None
    def predict_frame(self, frame):
        if not self.loaded:
            print(f"Model not yet load, please load model and try again....")
            return []

        # Make predictions
        t1 = current_milli_time()
        print("-------->start predict, frame shape = {0}".format(frame.shape))
        results = self.model.predict(frame, save=False, imgsz=self.imgsz, conf=self.conf)
        print("--------->finnished, took {0} ms, number result = {1}".format(current_milli_time() - t1, len(results)))
        return self.get_4points(results, frame)


    def predict(self, source):
            if not self.loaded:
                print(f"Model not yet load, please load model and try again....")
                return None
            try:
                # Make predictions
                t1 = current_milli_time()
                print("--------->start predict {0}".format(source))
                img = cv2.imread(source)
                print("img.shape", img.shape)
                print("img.shape", type(img))
                results = self.model.predict(img, save=False, imgsz=self.imgsz, conf=self.conf)
                print("--------->finnished, take {0} ms, resutl = {1}".format(current_milli_time() - t1, len(results)))
                self.get_4points(results, img)
                return results
            except:
                print(f"Can't predict")
            return None


    def get_4points(self, results, img):
            print("results = ", results)

            masks = results[0].masks
            if masks is not None:
                n_points = masks.xy
            else:
                return []
            
            mask = np.array(n_points, np.int32)
            #print("mask",mask)
            #print("mask",mask.shape)
            

            if mask is not None:
                #drawing
                image_with_mask = img.copy()
                color = [random.randint(0, 255) for _ in range(3)]
                cv2.fillPoly(image_with_mask, pts=[mask], color=color)
                img = cv2.addWeighted(img, 0.5, image_with_mask, 0.5, 1)

                color = color or (255, 255, 255)
                tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
                rect = cv2.minAreaRect(mask)

                center, (width, height), angle = rect
                # Add padding (in pixels) to the width and height
                padding = -20  # Adjust this value as needed
                width += 2 * padding
                height += 2 * padding

                # Recreate the rectangle with padding
                rect_with_padding = ((center[0], center[1]), (width, height), angle)
                box_with_padding = cv2.boxPoints(rect_with_padding).astype(np.uintp)

                print("rect", rect)
                box = cv2.boxPoints(rect).astype(np.uintp)
                print("box", box)
                cv2.drawContours(img, [box], 0, color, tl)
                cv2.drawContours(img, [box_with_padding], 0, (0,255,0), tl)
                
                #save file
                self.saved_file_detector = f'frame/detector_{current_milli_time()}.png'
                cv2.imwrite(self.saved_file_detector, img)

                print("box get 3D", box_with_padding)
                return box_with_padding
            return []

