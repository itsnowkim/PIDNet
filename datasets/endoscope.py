# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch
from .base_dataset import BaseDataset

class Endoscope(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path,
                 num_classes=11,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=1920, # 긴 게 base size
                 crop_size=(1072, 1920), # height, width 동일하게 설정하면 no crop 으로 학습 가능
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(Endoscope, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()

        self.label_mapping = {-1: ignore_label, 0: 0, 
                              1: 1, 2:2, 3:3, 4:4, 5:5,
                              6:6, 7:7, 8:8, 9:9, 10:10 }
        self.class_index_dict = {0: 'Bone', 1: 'LF', 2: 'Vessel', 3: 'Fat',
                                 4: 'SoftTissue', 5: 'Disc', 6: 'Instrument',
                                 7: 'Cage', 8: 'Screw', 9: 'Care', 10: 'BF'}
        
        self.class_weights = None
        
        self.bd_dilate_size = bd_dilate_size
    
    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name
                })
        return files
        
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root,'endoscope',item["img"]),
                           cv2.IMREAD_COLOR)
        size = image.shape

        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        label = cv2.imread(os.path.join(self.root,'endoscope',item["label"]),
                           cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)

        # random crop 적용하는 부분
        image, label, edge = self.gen_custom_sample(image, label, 
                                self.multi_scale, self.flip, edge_size=self.bd_dilate_size)

        # import pdb; pdb.set_trace();
        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    
    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred


    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
        
