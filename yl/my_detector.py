from ultralytics import YOLO
from ultralytics import settings
from lib.mylib import*
from vision_dl.drawing import*
from yl.detector_param import*

import numpy as np
import cv2
import random



class My_Detector:
    def __init__(self, detector_param:Detector_Param) -> None:
        self.param = detector_param
        self.loaded = False
        self.saved_file_detector = "logo.jpg"
        self.label_map = load_labels(detector_param.label_path)
        self.load_model()
    def load_model(self):
        try:
            # Load a model
            t1 = current_milli_time()
            print(f"Begin load model {self.param.model_path}...")
            self.model = YOLO(self.param.model_path)  # pretrained YOLOv8n model
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
            results = self.model.predict(source, save=self.param.saved_img, imgsz=self.param.imgsz, conf=self.param.conf)
            print("--------->finnished, take {0} ms".format(current_milli_time() - t1))
            print(results[0])
            return results
        except:
            print(f"Can't predict")
        return None
    def predict_frame2(self, frame):
        if not self.loaded:
            print(f"Model not yet load, please load model and try again....")
            return np.array([])

        # Make predictions
        t1 = current_milli_time()
        # print("-------->start predict, frame shape = {0}".format(frame.shape))
        results = self.model.predict(frame, save=self.param.saved_img, imgsz=self.param.imgsz, conf=self.param.conf)
        # print("--------->finnished, took {0} ms, number result = {1}".format(current_milli_time() - t1, len(results)))
        #debug result
        # print("====================Debug result tuanna=============")
        for result in results:
            boxes = result.boxes
            for index, box in enumerate(boxes):
                print(f'index = {index}, box = {box}')

            masks = result.masks
            for index, mask in enumerate(masks):
                # print(f'index = {index}, masks = {mask}')
                pass

            # keypoints = result.keypoints
            names = result.names
            for index, name in enumerate(names):
                print(f'index = {index}, name = {name}')

        print("====================Debug result=============")
        label_map = {0: 'Cat', 1: 'Dog', 2: 'Bird'}
        nc = len(label_map)
        colors = np.random.uniform(0, 255, size=(nc, 3))
        mat = plot_results(results, frame, label_map=label_map, colors=colors)
        b = cv2.imwrite("abc.png", mat)
        print("====================drawing done============={0}".format(b))
        #end debug
        #return self.get_4points(results, frame)
        return np.array([])

    def predict_frame(self, frame):
        if not self.loaded:
            print(f"Model not yet load, please load model and try again....")
            return np.array([])

        # Make predictions
        t1 = current_milli_time()
        # print("-------->start predict, frame shape = {0}".format(frame.shape))
        result = self.model.predict(frame, save=self.param.saved_img, imgsz=self.param.imgsz, conf=self.param.conf)[0]
        # print("--------->finnished, took {0} ms, number result = {1}".format(current_milli_time() - t1, len(result)))
        #debug result
        # print("====================Debug result tuanna=============")
        boxes = result.boxes
        masks = result.masks
        
        n_points = n_bndBox = n_class_index = n_conf = []

        if masks is not None:
            n_points = masks.xy

        if len(boxes):
            n_bndBox = boxes.xyxy.cpu().numpy().astype(np.uintp)
            n_class_index = list(map(int, boxes.cls))
            n_conf = list(map(float, boxes.conf))

        results = []
        for i in range(len(n_bndBox)):
            mask = n_points[i].astype(np.uintp) if masks is not None else None

            rect = cv2.minAreaRect(mask) if mask is not None else None
            # print(f'rect origin = {rect}')
            center, (width, height), angle = rect

            #rect offet
            width += 2 * self.param.offset_width
            height += 2 * self.param.offset_height
            rect_offset = ((center[0], center[1]), (width, height), angle)
            # print(f'rect offset origin = {rect_offset}')
            #rect dimension
            rect_dim = [0.0, 0.0, 0.0]

            result = DNNRESULT(
                class_index=n_class_index[i],
                box=n_bndBox[i],
                mask=mask,
                conf=n_conf[i],
                rect=rect,
                rect_offset=rect_offset,
                rect_dim=rect_dim
            )
            # print(f'result origin = {result}')

            results.append(result)

        results = sorted(results, key=lambda x: x.conf, reverse=True)
        # print("====================Debug result=============")
        # label_map = self.label_map
        # nc = len(label_map)
        # colors = np.random.uniform(0, 255, size=(nc, 3))
        # mat = plot_results(results, frame, label_map=label_map, colors=colors)
        # b = cv2.imwrite("abc.png", mat)
        # print("====================drawing============={0}".format(b))
        #end debug
        #return self.get_4points(results, frame)
        return results


    def predict_source(self, source):
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
                results = self.model.predict(img, save=self.param.saved_img, imgsz=self.param.imgsz, conf=self.param.conf)
                print("--------->finnished, take {0} ms, resutl = {1}".format(current_milli_time() - t1, len(results)))
                
                #debug result
                print("====================Debug result=============")
                for result in results:
                    boxes = result.boxes
                    masks = result.masks
                    keypoints = result.keypoints
                    probs = result.probs

                    print(f'boxes = {boxes}')
                    print(f'masks = {masks}')
                    print(f'keypoints = {keypoints}')
                    print(f'probs = {probs}')
                print("====================Debug result=============")
                #end debug
                
                # return self.get_4points(results, img)
                return np.array([])
            except:
                print(f"Can't predict")
            return None


    def get_4points(self, results, img):
            print("results = ", results)

            masks = results[0].masks
            if masks is not None:
                n_points = masks.xy
            else:
                return np.array([])
            
            mask = np.array(n_points, np.int32)
            #print("mask",mask)
            #print("mask",mask.shape)
            

            #if mask is not None:
            if mask.size != 0:
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
                #padding = -20  # Adjust this value as needed

                width += 2 * self.param.offset_width
                height += 2 * self.param.offset_height

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
                is_saved_file = cv2.imwrite(self.saved_file_detector, img)

                print("box get 3D = {0}, save file = {1}".format(box_with_padding, is_saved_file))
                return box_with_padding
            else:
                #save file empty
                #save file
                self.saved_file_detector = f'frame/can_not_detector_{current_milli_time()}.png'
                is_saved_file = cv2.imwrite(self.saved_file_detector, img)
                print("saved file {0}= {1}".format(self.saved_file_detector, is_saved_file))
            return np.array([])

