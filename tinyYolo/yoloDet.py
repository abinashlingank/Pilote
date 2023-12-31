import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import random
import ctypes
import pycuda.driver as cuda
import time

from exif_details import  get_width_height
from lat_long import get_coord
from efif_lat import get_info
import math
from gsd_cal import get_gsd
import json


EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
host_inputs  = []
cuda_inputs  = []
host_outputs = []
cuda_outputs = []
bindings = []


class YoloTRT():
    def __init__(self, library, engine, conf, yolo_ver):
        self.CONF_THRESH = conf 
        self.IOU_THRESHOLD = 0.4
        self.LEN_ALL_RESULT = 38001
        self.LEN_ONE_RESULT = 38
        self.yolo_version = yolo_ver
        self.categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"]
        
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)

        ctypes.CDLL(library)

        with open(engine, 'rb') as f:
            serialized_engine = f.read()

        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.batch_size = self.engine.max_batch_size

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.input_w = self.engine.get_binding_shape(binding)[-1]
                self.input_h = self.engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

    def PreProcessImg(self, img):
        image_raw = img
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        image = cv2.resize(image, (tw, th))
        image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128))
        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def Inference(self, img):
        input_image, image_raw, origin_h, origin_w = self.PreProcessImg(img)
        np.copyto(host_inputs[0], input_image.ravel())
        stream = cuda.Stream()
        self.context = self.engine.create_execution_context()
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        t1 = time.time()
        self.context.execute_async(self.batch_size, bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()
        t2 = time.time()
        output = host_outputs[0]
                
        for i in range(self.batch_size):
            result_boxes, result_scores, result_classid = self.PostProcess(output[i * self.LEN_ALL_RESULT: (i + 1) * self.LEN_ALL_RESULT], origin_h, origin_w)
        img_coor = []
        det_res = []
        act_coor = []
        for j in range(len(result_boxes)):
            box = result_boxes[j]
            det = dict()
            det["class"] = self.categories[int(result_classid[j])]
            det["conf"] = result_scores[j]
            det["box"] = box 
            det_res.append(det)
            self.PlotBbox(box, img, label="{}:{:.2f}".format(self.categories[int(result_classid[j])], result_scores[j]),)
        for i in det_res:
            img_coor.append(i['box'])
        for i in img_coor:
            act_coor.append((i[-2], i[-1]))

        path = r'test.jpg'
        photo_dimensions = get_width_height(path)
        # print(photo_dimensions)
        tda = {}
        middle_photo = (photo_dimensions[0]//2,photo_dimensions[1]//2)
        for i in range(0, len(act_coor)):
            
            middle_box = (float(act_coor[i][0]), float(act_coor[i][1]))
            
            # print(middle_box)

            
            distance = (((middle_box[0]-middle_photo[0])**2)+((middle_box[1]-middle_photo[1])**2))**(0.5)

            # gsd = get_gsd(path, s_width = 6.97, s_height = 3.92)cls


            d = 0.11*distance

            d = d/100000

            y_distance = (((middle_box[0]-middle_photo[0])**2))**(0.5)
            # print(distance, y_distance)
            #pythagoras theorem
            # x_distance = ((distance**2) - (y_distance**2))**(0.5)
            angle = np.arcsin(y_distance/distance) 
            '''USE A BETTER WAY TO FIND ANGLE'''################################
            angle = math.degrees(angle)

            dis_lst = []
            ph_end_x, ph_end_y = photo_dimensions[0], photo_dimensions[1]
            # print((ph_end_x, ph_end_y))

            # print (middle_box)


            quad_1_dis = (((middle_box[0]-ph_end_x)**2)+((middle_box[1])**2))**(0.5)
            quad_2_dis = (((middle_box[0])**2)+((middle_box[1])**2))**(0.5)
            quad_3_dis = (((middle_box[0])**2)+((middle_box[1]-ph_end_y)**2))**(0.5)
            quad_4_dis = (((middle_box[0]-ph_end_x)**2)+((middle_box[1]-ph_end_y)**2))**(0.5)


            dis_lst.append(quad_1_dis)
            dis_lst.append(quad_2_dis)
            dis_lst.append(quad_3_dis)
            dis_lst.append(quad_4_dis)

            up_angle = 0
            quad = ''

            if min(dis_lst) == quad_1_dis:
                up_angle = angle 
                quad = 'Quadrant 1'   
            elif min(dis_lst) == quad_2_dis:      
                up_angle = -angle
                quad = 'Quadrant 2'
            elif min(dis_lst) == quad_3_dis:   
                up_angle = -(90+angle)
                quad = 'Quadrant 3'
            elif min(dis_lst) == quad_4_dis:
                up_angle = 90+angle
                quad = 'Quadrant 4'
            
            
            

            cr = get_info(path)

            lat, long = get_coord(12.971869, 80.042952, up_angle, d, R=6378.137)
            coorr = (lat, long)

            tda[f'object {i}'] = coorr
            ######## get_coord(lat1, lon1, bearing, d, R=6378.137)

        corner_pix = ((middle_photo[0]**2)+middle_photo[1]**2)**0.5
            
        corner_dis = 0.11*corner_pix
        corner_dis = corner_dis/100000
        # print(corner_dis)
        


        # print(corner_dis, botttom_right_dis)
        # top_left_coor = get_coord(12.971869, 80.042952,-45 , corner_dis, R=6378.137)
        # print('Top Left: ',(top_left_coor))
        
        
        # bottom_left = get_coord(12.971869, 80.042952, -135,corner_dis, R=6378.137)
        # print('Bottom Left: ',(bottom_left))
        
        
        # bottom_right_coor = get_coord(12.971869, 80.042952, 135,corner_dis, R=6378.137)
        # print('Bottom Right: ',(bottom_right_coor))

        # top_right = get_coord(12.971869, 80.042952, 45,corner_dis, R=6378.137)
        # print('Top Right: ',(top_right))


            

        

        # LOGGER.info(f"{contentss, distance, y_distance, middle_box, middle_photo, angle}{'' if len(det) else '(no detections), '}")
    # print(img_lst)
        ff = open(f'{path}.json', 'w')



        ff.write(json.dumps(tda))
        ff.close()





        return det_res, t2-t1

    

    def PostProcess(self, output, origin_h, origin_w):
        num = int(output[0])
        if self.yolo_version == "v5":
            pred = np.reshape(output[1:], (-1, self.LEN_ONE_RESULT))[:num, :]
            pred = pred[:, :6]
        elif self.yolo_version == "v7":
            pred = np.reshape(output[1:], (-1, 6))[:num, :]
        
        boxes = self.NonMaxSuppression(pred, origin_h, origin_w, conf_thres=self.CONF_THRESH, nms_thres=self.IOU_THRESHOLD)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid
    
    def NonMaxSuppression(self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
        boxes = prediction[prediction[:, 4] >= conf_thres]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w -1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w -1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h -1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h -1)
        confs = boxes[:, 4]
        boxes = boxes[np.argsort(-confs)]
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes
    
    def xywh2xyxy(self, origin_h, origin_w, x):
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h
        return y
    
    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou
    
    def PlotBbox(self, x, img, color=None, label=None, line_thickness=None):
        tl = (line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA,)
        