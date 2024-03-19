# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------

import glob
import argparse
import cv2
import os
import numpy as np
import _init_paths
import models
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

color_map = [(128, 64,128),
             (244, 35,232),
             ( 70, 70, 70),
             (102,102,156),
             (190,153,153),
             (153,153,153),
             (250,170, 30),
             (220,220,  0),
             (107,142, 35),
             (152,251,152),
             ( 70,130,180),
             (220, 20, 60),
             (255,  0,  0),
             (  0,  0,142),
             (  0,  0, 70),
             (  0, 60,100),
             (  0, 80,100),
             (  0,  0,230),
             (119, 11, 32)]

# 각 클래스에 대한 색상을 정의 (R, G, B 형식)
class_colors = {
    'Bone': (153,153,153),  # 밝은 회색
    'LF': (107,142, 35),  # 올리브색
    'Vessel': (0, 255, 0),  # 초록색
    'Fat': (255, 255, 0),  # 노란색
    'SoftTissue': (70,130,180),  # 청록색
    'Dura' : (107, 102, 255), # 연한 보라색
    'Disc': (0, 0, 255),  # 파란색
    'Instrument': (119, 11, 32), # 어두운 자주색
    'Cage':(  0,  0, 70),  # 매우 어두운 파란색
    'Screw': (255, 165, 0),  # 주황색
    'Care': (64, 224, 208),  # 터콰이즈
    'BF':     (220, 20, 60),  # 밝은 빨간색
}

custom_color_map = [
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
]

def parse_resolution(s):
    try:
        width, height = map(int, s.split())
        return (width, height)
    except:
        raise argparse.ArgumentTypeError("Resolution must be width and height separated by space (e.g., '960 960')")

def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')
    parser.add_argument('--s', help='input image size [ex) 960 960]', default=(960, 960), type=parse_resolution)
    parser.add_argument('--custom', help='custom dataset or not', type=bool, default=True)
    parser.add_argument('--a', help='pidnet-s, pidnet-m or pidnet-l', default='pidnet-s', type=str)
    parser.add_argument('--c', help='cityscapes pretrained or not', type=bool, default=True)
    parser.add_argument('--p', help='dir for pretrained model', default='../pretrained_models/cityscapes/PIDNet_L_Cityscapes_test.pt', type=str)
    parser.add_argument('--r', help='root or dir for input images', default='samples/custom/', type=str)
    parser.add_argument('--t', help='the format of input images (.jpg, .png, ...)', default='.png', type=str)     

    args = parser.parse_args()

    return args

def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image

def load_pretrained2(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    
    # msg = 'Loaded {} parameters!'.format(len(pretrained_dict))

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model

def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict = False)
    
    return model

# 원본 이미지와 인퍼런스 결과를 혼합하기 위해 알파 블렌딩을 사용
def blend_image_with_mask(original_img, mask, color_palete, alpha=0.5, gamma=0):
    # mask는 (height, width)의 2D 배열이고, 각 픽셀의 값은 해당 클래스의 인덱스입니다.
    # custom_color_map을 사용하여 mask에 색을 적용합니다.
    beta = 1 - alpha
    # print(original_img.shape, mask.shape)
    color_mask = np.zeros_like(original_img)
    for i, color in enumerate(color_palete):
        color_mask[mask == i] = color
    
    # 원본 이미지와 색상 마스크를 혼합합니다.
    blended_img = cv2.addWeighted(original_img, beta, color_mask, alpha, gamma)
    return blended_img

if __name__ == '__main__':
    args = parse_args()
    images_list = glob.glob(args.r+'*'+args.t)
    print(args.r, args.t)
    print(f'len(images_list) : {len(images_list)}')
    sv_path = args.r+'outputs/'
    
    # get image size
    width, height = args.s
    if args.custom:
        # construct model
        model = models.pidnet.get_pred_model(args.a, 12, True) # 12 로 수정 필요
        # load model pretrained
        model = load_pretrained2(model, args.p).cuda()
    else:
        model = models.pidnet.get_pred_model(args.a, 19 if args.c else 11)
        model = load_pretrained(model, args.p).cuda()

    model.eval()
    with torch.no_grad():
        for img_path in tqdm(images_list):
            img_name = img_path.split("\\")[-1]
            original_img = cv2.imread(os.path.join(args.r, img_name),
                               cv2.IMREAD_COLOR)
            # image shape change
            img = cv2.resize(original_img, (width, height), interpolation=cv2.INTER_AREA)

            sv_img = np.zeros_like(img).astype(np.uint8)
            img = input_transform(img)
            img = img.transpose((2, 0, 1)).copy()
            img = torch.from_numpy(img).unsqueeze(0).cuda()
            pred = model(img)

            # pred size 맞춰주기
            pred = F.interpolate(pred, size=original_img.shape[:2], 
                                 mode='bilinear', align_corners=True)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            
            if args.custom:
                blended_img = blend_image_with_mask(original_img, pred, custom_color_map ,alpha=0.3)
            else:
                blended_img = blend_image_with_mask(original_img, pred, color_map ,alpha=0.8)
            
            # image 저장하기 전 처리
            blended_img = Image.fromarray(blended_img)
            
            if not os.path.exists(sv_path):
                os.mkdir(sv_path)
            blended_img.save(sv_path+img_name)
            
            
            
        
        