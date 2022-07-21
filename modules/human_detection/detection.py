import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Detection:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    def crop(self, bbox, image):
        bbox = self.remove_bbox_noise(bbox, image)
        for bb in bbox:
            return image[bb[1]:bb[3], bb[0]:bb[2]]

    def remove_bbox_noise(self, bbox, image, thresh=0.3):
        area_image = image.shape[0] * image.shape[1]
        new_bb = []
        for bb in bbox:
            x1, y1, x2, y2 = bb
            area_bbox = (bb[2] - bb[0]) * (bb[3] - bb[1])
            ratio = area_bbox / area_image
            if ratio > thresh:
                new_bb.append(bb)
        return new_bb

    def run(self, image):
        result = self.model(image)
        output = result.pandas().xyxy[0]
        bbox = np.int32(np.array(output)[:,:4][np.where(np.array(output)[:,6] == 'person')])
        image = self.crop(bbox, image)
        return image