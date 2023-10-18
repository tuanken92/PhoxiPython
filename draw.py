import cv2
import numpy as np
import random
import os
import time
from collections import namedtuple, deque

print("OpenCV: ", cv2.__version__)


DNNRESULT = namedtuple("DetResult", 
                       ["class_index", "box", "mask", "conf", "rect"], 
                       defaults=5*[None])

def plot_text(text, img:np.ndarray, org:tuple=None, color:tuple=None, line_thickness=5):
    """
    Helper function for drawing single min area rect on image
    Parameters:
        text : string
        img (np.ndarray): input image
    """
    color = color or (255, 255, 255)
    org = org or (10, 10)
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    tf = max(tl -1, 1)
    
    cv2.putText(img, text, org, 0, tl / 3, color, tf, cv2.LINE_AA)
    return img
    pass

def plot_one_min_rect(rect, img:np.ndarray, color:tuple=None, line_thickness=5):
    """
    Helper function for drawing single min area rect on image
    Parameters:
        rect :result of cv2.minAreaRect
        img (np.ndarray): input image
    """
    color = color or (255, 255, 255)
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    box = cv2.boxPoints(rect).astype(np.uint0)
    cv2.drawContours(img, [box], 0, color, tl)
    return img
    pass

def plot_one_box(box:np.ndarray, img:np.ndarray, color=None, mask:np.ndarray = None, label:str = None, line_thickness:int = 5):
        """
        Helper function for drawing single bounding box on image
        Parameters:
            x (np.ndarray): bounding box coordinates in format [x1, y1, x2, y2]
            img (no.ndarray): input image
            color (Tuple[int, int, int], *optional*, None): color in BGR format for drawing box, if not specified will be selected randomly
            mask (np.ndarray, *optional*, None): instance segmentation mask polygon in format [N, 2], where N - number of points in contour, if not provided, only box will be drawn
            label (str, *optonal*, None): box label string, if not provided will not be provided as drowing result
            line_thickness (int, *optional*, 5): thickness for box drawing lines
        """
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        fs = tl / 2.5
        if color is None:
            color = [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=fs, thickness=tf)[0]

            if c1[1] > t_size[1] + 3:
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                org = (c1[0], c1[1] -2)
            else:
                c2 = c1[0] + t_size[0], c1[1] + t_size[1] + 3
                org = (c1[0], c1[1] + t_size[1] + 2)

            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, org, 0, fs, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        if mask is not None:
            image_with_mask = img.copy()
            cv2.fillPoly(image_with_mask, pts=[mask], color=color)
            img = cv2.addWeighted(img, 0.5, image_with_mask, 0.5, 1)
        return img

def plot_results(results, img:np.ndarray, label_map={}, colors=[]):
        """
        Helper function for drawing bounding boxes on image
        Parameters:
            results: list DNNRESULT("class_index", "box", "mask", "conf")
            source_image (np.ndarray): input image for drawing
            label_map; (Dict[int, str]): label_id to class name mapping
        Returns:

        """
        result:DNNRESULT = None
        for result in results:
            box = result.box
            mask = result.mask
            rect = result.rect
            cls_index = result.class_index
            conf = result.conf

            h, w = img.shape[:2]

            if label_map:
                label = f'{label_map[cls_index]}, score: {conf:.2f}'
            else:
                label = f"OBJ, score: {conf:.2f}"
            if len(colors):
                color = colors[cls_index]
            else:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            img = plot_one_box(box, img, 
                                mask=mask, 
                                label=label, 
                                color=color, line_thickness=1)
            
            color = (0, 255, 255)
            if mask is not None:
                plot_one_min_rect(rect, img, 
                                color=color, 
                                line_thickness=1)
        return img

img = cv2.imread("test1.png")

data = [{'x': 82.25, 'y': 296.25}, {'x': 82.25, 'y': 448.75}, {'x': 85.75, 'y': 451.25}, {'x': 87.5, 'y': 451.25}, {'x': 89.25, 'y': 450.0}, {'x': 96.25, 'y': 450.0}, {'x': 98.0, 'y': 448.75}, {'x': 285.25, 'y': 448.75}, {'x': 287.0, 'y': 450.0}, {'x': 295.75, 'y': 450.0}, {'x': 297.5, 'y': 448.75}, {'x': 341.25, 'y': 448.75}, {'x': 343.0, 'y': 450.0}, {'x': 383.25, 'y': 450.0}, {'x': 385.0, 'y': 451.25}, {'x': 400.75, 'y': 451.25}, {'x': 402.5, 'y': 450.0}, {'x': 414.75, 'y': 450.0}, 
{'x': 416.5, 'y': 448.75}, {'x': 418.25, 'y': 450.0}, {'x': 493.5, 'y': 450.0}, {'x': 495.25, 'y': 451.25}, {'x': 535.5, 'y': 451.25}, {'x': 537.25, 'y': 452.5}, {'x': 603.75, 'y': 452.5}, {'x': 605.5, 'y': 451.25}, {'x': 607.25, 'y': 451.25}, {'x': 609.0, 'y': 450.0}, {'x': 612.5, 'y': 450.0}, {'x': 619.5, 'y': 445.0}, {'x': 619.5, 'y': 296.25}, {'x': 451.5, 'y': 296.25}, {'x': 449.75, 'y': 297.5}, {'x': 427.0, 'y': 297.5}, {'x': 425.25, 'y': 296.25}, {'x': 421.75, 'y': 296.25}, {'x': 420.0, 'y': 297.5}, {'x': 409.5, 'y': 297.5}, {'x': 407.75, 'y': 298.75}, {'x': 397.25, 'y': 298.75}, {'x': 395.5, 'y': 297.5}, {'x': 392.0, 'y': 297.5}, {'x': 390.25, 'y': 296.25}, {'x': 225.75, 'y': 296.25}, {'x': 224.0, 'y': 297.5}, {'x': 222.25, 'y': 297.5}, {'x': 220.5, 'y': 298.75}, {'x': 211.75, 'y': 298.75}, {'x': 210.0, 'y': 297.5}, {'x': 204.75, 'y': 297.5}, {'x': 203.0, 'y': 296.25}, {'x': 194.25, 'y': 296.25}, {'x': 192.5, 'y': 297.5}, {'x': 182.0, 'y': 297.5}, {'x': 180.25, 'y': 298.75}, {'x': 161.0, 'y': 298.75}, {'x': 
159.25, 'y': 297.5}, {'x': 155.75, 'y': 297.5}, {'x': 154.0, 'y': 296.25}]

data = [{'x': 78.75, 'y': 425.625}, {'x': 78.75, 'y': 652.5}, {'x': 86.625, 'y': 652.5}, {'x': 89.25, 'y': 650.625}, {'x': 115.5, 'y': 650.625}, {'x': 118.125, 'y': 652.5}, {'x': 199.5, 'y': 652.5}, {'x': 202.125, 'y': 654.375}, {'x': 254.625, 'y': 654.375}, {'x': 257.25, 'y': 652.5}, {'x': 259.875, 'y': 652.5}, {'x': 262.5, 'y': 654.375}, {'x': 265.125, 'y': 654.375}, {'x': 267.75, 'y': 652.5}, {'x': 294.0, 'y': 652.5}, {'x': 296.625, 'y': 650.625}, {'x': 391.125, 'y': 650.625}, {'x': 393.75, 'y': 652.5}, {'x': 425.25, 'y': 652.5}, {'x': 427.875, 'y': 654.375}, {'x': 433.125, 'y': 654.375}, {'x': 435.75, 'y': 656.25}, {'x': 438.375, 'y': 654.375}, {'x': 485.625, 'y': 654.375}, {'x': 488.25, 'y': 656.25}, {'x': 577.5, 'y': 656.25}, {'x': 580.125, 'y': 654.375}, {'x': 588.0, 'y': 654.375}, {'x': 590.625, 'y': 652.5}, {'x': 658.875, 'y': 652.5}, {'x': 661.5, 'y': 650.625}, {'x': 679.875, 'y': 650.625}, {'x': 682.5, 'y': 652.5}, {'x': 724.5, 'y': 652.5}, {'x': 727.125, 'y': 654.375}, {'x': 761.25, 'y': 654.375}, {'x': 763.875, 'y': 656.25}, {'x': 863.625, 'y': 656.25}, {'x': 866.25, 'y': 654.375}, {'x': 868.875, 'y': 654.375}, {'x': 879.375, 'y': 646.875}, {'x': 879.375, 'y': 519.375}, {'x': 882.0, 'y': 517.5}, {'x': 882.0, 'y': 498.75}, {'x': 884.625, 'y': 496.875}, {'x': 884.625, 'y': 442.5}, {'x': 882.0, 'y': 440.625}, {'x': 882.0, 'y': 429.375}, {'x': 879.375, 'y': 427.5}, {'x': 879.375, 'y': 425.625}, {'x': 698.25, 'y': 425.625}, {'x': 695.625, 'y': 427.5}, {'x': 609.0, 'y': 427.5}, {'x': 606.375, 'y': 425.625}]

data = [{'x': 593.25, 'y': 262.5}, {'x': 591.5, 'y': 263.75}, {'x': 588.0, 'y': 263.75}, {'x': 586.25, 'y': 265.0}, {'x': 574.0, 'y': 265.0}, {'x': 572.25, 'y': 266.25}, {'x': 570.5, 'y': 266.25}, {'x': 567.0, 'y': 268.75}, {'x': 563.5, 'y': 268.75}, {'x': 561.75, 'y': 270.0}, {'x': 
547.75, 'y': 270.0}, {'x': 546.0, 'y': 271.25}, {'x': 542.5, 'y': 271.25}, {'x': 540.75, 'y': 272.5}, {'x': 539.0, 'y': 272.5}, {'x': 537.25, 'y': 273.75}, {'x': 532.0, 'y': 273.75}, {'x': 530.25, 'y': 275.0}, {'x': 514.5, 'y': 275.0}, {'x': 512.75, 'y': 276.25}, {'x': 509.25, 'y': 276.25}, {'x': 505.75, 'y': 278.75}, {'x': 502.25, 'y': 278.75}, {'x': 500.5, 'y': 280.0}, {'x': 491.75, 'y': 280.0}, {'x': 490.0, 'y': 281.25}, {'x': 488.25, 'y': 281.25}, {'x': 486.5, 'y': 282.5}, {'x': 484.75, 'y': 282.5}, {'x': 483.0, 'y': 283.75}, {'x': 479.5, 'y': 283.75}, {'x': 477.75, 'y': 285.0}, {'x': 463.75, 'y': 285.0}, {'x': 462.0, 'y': 286.25}, {'x': 460.25, 'y': 286.25}, {'x': 458.5, 'y': 287.5}, {'x': 456.75, 'y': 287.5}, {'x': 455.0, 'y': 288.75}, {'x': 453.25, 'y': 288.75}, {'x': 451.5, 'y': 290.0}, {'x': 439.25, 'y': 290.0}, {'x': 437.5, 'y': 291.25}, {'x': 434.0, 'y': 291.25}, {'x': 430.5, 'y': 293.75}, {'x': 427.0, 'y': 293.75}, {'x': 425.25, 'y': 295.0}, {'x': 409.5, 'y': 295.0}, {'x': 407.75, 'y': 296.25}, {'x': 406.0, 'y': 296.25}, {'x': 404.25, 'y': 297.5}, {'x': 402.5, 'y': 297.5}, {'x': 400.75, 'y': 298.75}, {'x': 395.5, 'y': 298.75}, {'x': 393.75, 'y': 300.0}, {'x': 385.0, 'y': 300.0}, {'x': 383.25, 'y': 301.25}, {'x': 379.75, 'y': 301.25}, {'x': 376.25, 'y': 303.75}, {'x': 371.0, 'y': 303.75}, {'x': 369.25, 'y': 305.0}, {'x': 351.75, 'y': 305.0}, {'x': 350.0, 'y': 306.25}, {'x': 344.75, 'y': 306.25}, {'x': 341.25, 'y': 308.75}, {'x': 337.75, 'y': 308.75}, {'x': 336.0, 'y': 310.0}, {'x': 320.25, 'y': 310.0}, {'x': 318.5, 'y': 311.25}, {'x': 316.75, 'y': 311.25}, {'x': 315.0, 'y': 312.5}, {'x': 313.25, 'y': 312.5}, {'x': 311.5, 'y': 313.75}, {'x': 308.0, 'y': 313.75}, {'x': 306.25, 'y': 315.0}, {'x': 294.0, 'y': 315.0}, {'x': 292.25, 'y': 316.25}, {'x': 288.75, 'y': 316.25}, {'x': 285.25, 'y': 318.75}, {'x': 280.0, 'y': 318.75}, {'x': 278.25, 'y': 320.0}, {'x': 267.75, 'y': 320.0}, {'x': 264.25, 'y': 322.5}, {'x': 262.5, 'y': 322.5}, {'x': 260.75, 'y': 323.75}, {'x': 259.0, 'y': 323.75}, {'x': 257.25, 'y': 325.0}, {'x': 
241.5, 'y': 325.0}, {'x': 239.75, 'y': 326.25}, {'x': 238.0, 'y': 326.25}, {'x': 234.5, 'y': 328.75}, {'x': 231.0, 'y': 328.75}, {'x': 229.25, 'y': 330.0}, {'x': 217.0, 'y': 330.0}, {'x': 215.25, 'y': 331.25}, {'x': 213.5, 'y': 331.25}, {'x': 211.75, 'y': 332.5}, {'x': 210.0, 'y': 332.5}, {'x': 208.25, 'y': 333.75}, {'x': 206.5, 'y': 333.75}, {'x': 204.75, 'y': 335.0}, {'x': 185.5, 'y': 335.0}, {'x': 183.75, 
'y': 336.25}, {'x': 182.0, 'y': 336.25}, {'x': 178.5, 'y': 338.75}, {'x': 176.75, 'y': 338.75}, {'x': 175.0, 'y': 340.0}, {'x': 162.75, 'y': 340.0}, {'x': 161.0, 'y': 341.25}, {'x': 157.5, 'y': 341.25}, {'x': 154.0, 'y': 343.75}, {'x': 152.25, 'y': 343.75}, {'x': 150.5, 'y': 345.0}, {'x': 131.25, 'y': 345.0}, {'x': 129.5, 'y': 346.25}, {'x': 127.75, 'y': 346.25}, {'x': 124.25, 'y': 348.75}, {'x': 122.5, 'y': 
348.75}, {'x': 120.75, 'y': 350.0}, {'x': 103.25, 'y': 350.0}, {'x': 101.5, 'y': 351.25}, {'x': 98.0, 'y': 351.25}, {'x': 94.5, 'y': 353.75}, {'x': 92.75, 'y': 353.75}, {'x': 91.0, 'y': 355.0}, {'x': 73.5, 
'y': 355.0}, {'x': 73.5, 'y': 395.0}, {'x': 75.25, 'y': 396.25}, {'x': 75.25, 'y': 397.5}, {'x': 77.0, 'y': 398.75}, {'x': 77.0, 'y': 417.5}, {'x': 78.75, 'y': 418.75}, {'x': 78.75, 'y': 421.25}, {'x': 80.5, 'y': 422.5}, {'x': 80.5, 'y': 425.0}, {'x': 82.25, 'y': 426.25}, {'x': 82.25, 'y': 431.25}, {'x': 84.0, 'y': 432.5}, {'x': 84.0, 'y': 451.25}, {'x': 85.75, 'y': 452.5}, {'x': 85.75, 'y': 455.0}, {'x': 87.5, 'y': 456.25}, {'x': 87.5, 'y': 458.75}, {'x': 89.25, 'y': 460.0}, {'x': 89.25, 'y': 471.25}, {'x': 91.0, 'y': 472.5}, {'x': 91.0, 'y': 490.0}, {'x': 92.75, 'y': 491.25}, {'x': 92.75, 'y': 496.25}, {'x': 
94.5, 'y': 497.5}, {'x': 94.5, 'y': 498.75}, {'x': 96.25, 'y': 500.0}, {'x': 96.25, 'y': 501.25}, {'x': 99.75, 'y': 503.75}, {'x': 122.5, 'y': 503.75}, {'x': 124.25, 'y': 502.5}, {'x': 126.0, 'y': 502.5}, {'x': 129.5, 'y': 500.0}, {'x': 131.25, 'y': 500.0}, {'x': 133.0, 'y': 498.75}, {'x': 148.75, 'y': 498.75}, {'x': 150.5, 'y': 497.5}, {'x': 152.25, 'y': 497.5}, {'x': 155.75, 'y': 495.0}, {'x': 157.5, 'y': 495.0}, {'x': 159.25, 'y': 493.75}, {'x': 183.75, 'y': 493.75}, {'x': 185.5, 'y': 492.5}, {'x': 187.25, 'y': 492.5}, {'x': 187.25, 'y': 491.25}, {'x': 189.0, 'y': 490.0}, {'x': 190.75, 'y': 490.0}, {'x': 192.5, 'y': 488.75}, {'x': 201.25, 'y': 488.75}, {'x': 203.0, 'y': 487.5}, {'x': 204.75, 'y': 487.5}, {'x': 208.25, 'y': 485.0}, {'x': 211.75, 'y': 485.0}, {'x': 213.5, 'y': 483.75}, {'x': 239.75, 'y': 483.75}, {'x': 241.5, 'y': 482.5}, {'x': 243.25, 'y': 482.5}, {'x': 246.75, 'y': 480.0}, {'x': 252.0, 'y': 480.0}, {'x': 253.75, 'y': 478.75}, {'x': 271.25, 'y': 478.75}, {'x': 273.0, 'y': 477.5}, {'x': 274.75, 'y': 477.5}, {'x': 276.5, 'y': 476.25}, {'x': 278.25, 'y': 476.25}, {'x': 280.0, 'y': 475.0}, {'x': 281.75, 'y': 475.0}, {'x': 283.5, 'y': 473.75}, {'x': 299.25, 'y': 473.75}, {'x': 301.0, 'y': 472.5}, {'x': 
304.5, 'y': 472.5}, {'x': 308.0, 'y': 470.0}, {'x': 309.75, 'y': 470.0}, {'x': 311.5, 'y': 468.75}, {'x': 325.5, 'y': 468.75}, {'x': 330.75, 'y': 465.0}, {'x': 332.5, 'y': 465.0}, {'x': 334.25, 'y': 463.75}, {'x': 348.25, 'y': 463.75}, {'x': 350.0, 'y': 462.5}, {'x': 353.5, 'y': 462.5}, {'x': 355.25, 'y': 461.25}, {'x': 357.0, 'y': 461.25}, {'x': 358.75, 'y': 460.0}, {'x': 362.25, 'y': 460.0}, {'x': 364.0, 'y': 458.75}, {'x': 378.0, 'y': 458.75}, {'x': 379.75, 'y': 457.5}, {'x': 381.5, 'y': 457.5}, {'x': 386.75, 'y': 453.75}, {'x': 407.75, 'y': 453.75}, {'x': 409.5, 'y': 452.5}, {'x': 411.25, 'y': 452.5}, {'x': 414.75, 'y': 450.0}, {'x': 420.0, 'y': 450.0}, {'x': 421.75, 'y': 448.75}, {'x': 439.25, 'y': 448.75}, {'x': 441.0, 'y': 447.5}, {'x': 442.75, 'y': 447.5}, {'x': 446.25, 'y': 445.0}, {'x': 448.0, 'y': 445.0}, {'x': 449.75, 'y': 443.75}, {'x': 472.5, 'y': 443.75}, {'x': 474.25, 'y': 442.5}, {'x': 476.0, 'y': 442.5}, {'x': 479.5, 'y': 440.0}, {'x': 483.0, 'y': 440.0}, {'x': 484.75, 'y': 438.75}, {'x': 498.75, 
'y': 438.75}, {'x': 500.5, 'y': 437.5}, {'x': 502.25, 'y': 437.5}, {'x': 504.0, 'y': 436.25}, {'x': 505.75, 'y': 436.25}, {'x': 507.5, 'y': 435.0}, {'x': 511.0, 'y': 435.0}, {'x': 512.75, 'y': 433.75}, {'x': 532.0, 'y': 433.75}, {'x': 533.75, 'y': 432.5}, {'x': 535.5, 'y': 432.5}, {'x': 537.25, 'y': 431.25}, {'x': 539.0, 'y': 431.25}, {'x': 540.75, 'y': 430.0}, {'x': 542.5, 'y': 430.0}, {'x': 544.25, 'y': 428.75}, {'x': 560.0, 'y': 428.75}, {'x': 561.75, 'y': 427.5}, {'x': 563.5, 'y': 427.5}, {'x': 565.25, 'y': 426.25}, {'x': 567.0, 'y': 426.25}, {'x': 568.75, 'y': 425.0}, {'x': 570.5, 'y': 425.0}, {'x': 572.25, 'y': 423.75}, {'x': 593.25, 'y': 423.75}, {'x': 595.0, 'y': 422.5}, {'x': 598.5, 'y': 422.5}, {'x': 602.0, 'y': 420.0}, {'x': 603.75, 'y': 420.0}, {'x': 605.5, 'y': 418.75}, {'x': 617.75, 'y': 418.75}, {'x': 619.5, 'y': 417.5}, {'x': 621.25, 'y': 417.5}, {'x': 623.0, 'y': 416.25}, {'x': 623.0, 'y': 377.5}, {'x': 621.25, 'y': 376.25}, {'x': 621.25, 'y': 346.25}, {'x': 619.5, 'y': 345.0}, {'x': 619.5, 'y': 340.0}, {'x': 617.75, 'y': 338.75}, {'x': 617.75, 'y': 336.25}, {'x': 616.0, 'y': 335.0}, {'x': 616.0, 'y': 327.5}, {'x': 614.25, 'y': 326.25}, {'x': 614.25, 'y': 300.0}, {'x': 612.5, 'y': 298.75}, {'x': 612.5, 'y': 293.75}, {'x': 610.75, 'y': 292.5}, {'x': 610.75, 'y': 291.25}, {'x': 609.0, 'y': 290.0}, {'x': 609.0, 'y': 287.5}, {'x': 607.25, 'y': 286.25}, {'x': 607.25, 'y': 267.5}, {'x': 605.5, 'y': 266.25}, 
{'x': 605.5, 'y': 265.0}, {'x': 603.75, 'y': 265.0}, {'x': 600.25, 'y': 262.5}]

# Extract 'x' and 'y' values into separate lists
x_values = [point['x'] for point in data]
y_values = [point['y'] for point in data]

# Create NumPy arrays
x_array = np.array(x_values)
y_array = np.array(y_values)

# Combine 'x' and 'y' arrays into a single NumPy array if needed
combined_array = np.column_stack((x_array, y_array))

# Print the arrays
print("X Values:")
print(x_array)
print("\nY Values:")
print(y_array)


# Extract 'x' and 'y' values into separate lists
x_values = [point['x'] for point in data]
y_values = [point['y'] for point in data]

# Create NumPy arrays
x_array = np.array(x_values, np.int32)
y_array = np.array(y_values, np.int32)

# Combine 'x' and 'y' arrays into a single NumPy array if needed
mask = np.column_stack((x_array, y_array))

if mask is not None:
    image_with_mask = img.copy()
    color = [random.randint(0, 255) for _ in range(3)]
    cv2.fillPoly(image_with_mask, pts=[mask], color=color)
    img = cv2.addWeighted(img, 0.5, image_with_mask, 0.5, 1)

    color = color or (255, 255, 255)
    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    rect = cv2.minAreaRect(mask)

    center, (width, height), angle = rect
    # Add padding (in pixels) to the width and height
    padding = -10  # Adjust this value as needed
    width += 2 * padding
    height += 2 * padding

    # Recreate the rectangle with padding
    rect_with_padding = ((center[0], center[1]), (width, height), angle)
    box_with_padding = cv2.boxPoints(rect_with_padding).astype(np.uint0)

    print("rect", rect)
    box = cv2.boxPoints(rect).astype(np.uint0)
    print("box", box)
    cv2.drawContours(img, [box], 0, color, tl)
    cv2.drawContours(img, [box_with_padding], 0, (0,255,0), tl)
    cv2.imwrite("hehe.png", img)
