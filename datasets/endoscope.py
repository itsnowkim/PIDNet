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
                 num_classes=13, # 총 12개 class, 1개는 background
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=1920, # 긴 게 base size
                 crop_size=(960, 960), # height, width 동일하게 설정하면 no crop 으로 학습 가능
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

        # 255 배경
        self.label_mapping = {0: 0, 
                              1: 1, 2:2, 3:3, 4:4, 5:5, 6:6,
                              7:7, 8:8, 9:9, 10:10, 11: 11,
                              255:12
                              }
        self.class_index_dict = {
            0: 'Bone', 1: 'LF', 2: 'Vessel', 3: 'Fat',
            4: 'SoftTissue', 5: 'Dura', 6: 'Disc',
            7: 'Instrument', 8: 'Cage', 9:'Screw', 10: 'Care', 11: 'BF',
            12: 'Background'
        }
        self.custom_color_map = [
            (153,153,153),  # 밝은 회색
            (107,142, 35),  # 올리브색
            (0, 255, 0),  # 초록색
            (255, 255, 0),  # 노란색
            (70,130,180),  # 청록색
            (107, 102, 255), # 연한 보라색
            (0, 0, 255),  # 파란색
            (119, 11, 32), # 어두운 자주색
            (  0,  0, 70),  # 매우 어두운 파란색
            (255, 165, 0),  # 주황색
            (64, 224, 208),  # 터콰이즈
            (220, 20, 60),  # 밝은 빨간색
            (0,0,0) # 검정색
        ]

        # self.label_mapping = {-1: ignore_label, 0: ignore_label, 
        #                       1: 1, 2:2, 3:3, 4:4, 5:5, 6:6,
        #                       7:7, 8:8, 9:9, 10:10, 11: 11, 12:12}
        # self.class_index_dict = {
        #     1: 'Bone', 2: 'LF', 3: 'Vessel', 4: 'Fat',
        #     5: 'SoftTissue', 6: 'Dura', 7: 'Disc',
        #     8: 'Instrument', 9: 'Cage', 10: 'Screw', 11: 'Care', 12: 'BF'
        # }
        
        # class weight 정의 - distribution 에 의해 정의?
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
        
    def convert_label(self, label):
        temp = label.copy()
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
        for k in range(preds.shape[0]):
            pred = self.convert_label(preds[k])
            sv_img = np.zeros((1080, 1920, 3), dtype=np.uint8)

            # 색상 적용
            for i, color in enumerate(self.custom_color_map):
                for j in range(3):
                    sv_img[:,:,j][pred==i] = self.custom_color_map[i][j]
            save_img = Image.fromarray(sv_img)
            # save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[k]+'.png'))
