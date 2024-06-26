# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import argparse
import os
import pprint

import logging
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

import _init_paths
import models
import datasets
import csv
from configs import config
from configs import update_config
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss
from utils.function import train, validate
from utils.utils import create_logger, FullModel
from utils.scheduler import CosineAnnealingWarmUpRestarts

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/cityscapes/pidnet_small_cityscapes.yaml",
                        type=str)
    parser.add_argument('--seed', type=int, default=304)    
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)        

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    if torch.cuda.device_count() != len(gpus):
        print("The gpu numbers do not match!")
        return 0
    
    imgnet = 'imagenet' in config.MODEL.PRETRAINED
    model = models.pidnet.get_seg_model(config, imgnet_pretrained=imgnet)
 
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)
    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])

    train_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TRAIN_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=config.TRAIN.MULTI_SCALE,
                        flip=config.TRAIN.FLIP,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TRAIN.BASE_SIZE,
                        crop_size=crop_size,
                        scale_factor=config.TRAIN.SCALE_FACTOR)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False,
        drop_last=True)
    

    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False)

    # criterion
    if config.LOSS.USE_OHEM:
        sem_criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                        thres=config.LOSS.OHEMTHRES,
                                        min_kept=config.LOSS.OHEMKEEP,
                                        weight=train_dataset.class_weights)
    else:
        sem_criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                    weight=train_dataset.class_weights)

    bd_criterion = BondaryLoss()
    
    model = FullModel(model, sem_criterion, bd_criterion)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # optimizer
    params_dict = dict(model.named_parameters())
    if config.TRAIN.SCHEDULER == 'cosinewarm':
        params = [{'params': list(params_dict.values()), 'lr': 0}]
    else:
        params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]
    if config.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(params,
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )
    elif config.TRAIN.OPTIMIZER == 'adam':
        if config.TRAIN.SCHEDULER == 'cosinewarm':
            optimizer = torch.optim.Adam(params, lr=0, betas=(0.9, 0.999), eps=1e-08, weight_decay=config.TRAIN.WD)
        else:
            optimizer = torch.optim.Adam(params, lr=config.TRAIN.LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=config.TRAIN.WD)
    elif config.TRAIN.OPTIMIZER == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=config.TRAIN.LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=config.TRAIN.WD)
    elif config.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=config.TRAIN.LR, alpha=0.99, eps=1e-08, weight_decay=config.TRAIN.WD, momentum=config.TRAIN.MOMENTUM)
    else:
        raise ValueError(f"Unsupported optimizer: {config.TRAIN.OPTIMIZER}")
    
    # scheduler
    # step - step size, gamma 설정 필요
    # cosine - tmax 설정 필요 50, 100, 200, 450
    if config.TRAIN.SCHEDULER == 'step':
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.SCHEDULER['STEP_SIZE'], gamma=config.SCHEDULER['GAMMA'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    elif config.TRAIN.SCHEDULER == 'cosine':
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.SCHEDULER['T_MAX'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    elif config.TRAIN.SCHEDULER == 'cosinewarm':
        # T_0:restart주기, T_mult=반복될 주기의 길이에 곱할 factor, eta_max=최대 lr,  T_up=warmup 에 필요한 epoch 수, gamma=진폭 곱)
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=30, T_mult=1, eta_max=config.TRAIN.LR,  T_up=10, gamma=0.5)
        print('scheduler : cosinewarm')
    else:
        raise ValueError(f"Unsupported scheduler type: {config.TRAIN.SCHEDULER}")

    epoch_iters = int(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))   
        
    best_mIoU = 0
    last_epoch = 0
    best_epoch = 0
    flag_rm = config.TRAIN.RESUME
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            best_epoch = checkpoint['best_epoch']
            dct = checkpoint['state_dict']
            
            model.module.model.load_state_dict({k.replace('model.', ''): v for k, v in dct.items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler 분기 처리?
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    real_end = 120+1 if 'camvid' in config.DATASET.TRAIN_SET else end_epoch
    
    # csv 설정
    csv_file = os.path.join(final_output_dir, 'results.csv')
    fieldnames = ['epoch', 'lr', 'train/acc', 'train/loss', 'train/sem_loss', 'train/bce_loss', 'train/sb_loss',
                  'val/acc', 'val/loss', 'val/sem_loss', 'val/bce_loss', 'val/sb_loss', 'mean_IoU',]
    
    # class IoU fieldname 추가
    for _, name in train_dataset.class_index_dict.items():
        fieldnames.append(name)

    # 파일이 존재하지 않으면 새로운 파일 생성 및 컬럼명 작성
    if not os.path.isfile(csv_file):
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    for epoch in range(last_epoch, real_end):

        current_trainloader = trainloader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)

        train_res_dict = train(config, epoch, config.TRAIN.END_EPOCH, 
                  epoch_iters, config.TRAIN.LR, num_iters,
                  trainloader, optimizer, model, writer_dict, scheduler)

        # validate in all epoch
        valid_loss, mean_IoU, IoU_array, val_res_dict = validate(config, testloader, model, writer_dict)
        if flag_rm == 1:
            flag_rm = 0

        logger.info('=> saving checkpoint to {}'.format(final_output_dir + 'checkpoint.pth.tar'))
        
        # csv 에 train_res, val_res 저장
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            result_row = {}
            # class 결과 먼저 작성
            for index, IoU_value in enumerate(IoU_array):
                class_name = train_dataset.class_index_dict.get(index, f"Unknown class index: {index}")
                result_row[class_name] = IoU_value
            
            # result row 에 train, val result 통합
            result_row.update(train_res_dict)
            result_row.update(val_res_dict)

            writer.writerow(result_row)

        # updaete - IoU 갱신
        if mean_IoU > best_mIoU:
            best_mIoU = mean_IoU
            best_epoch = epoch
            torch.save(model.module.state_dict(),
                    os.path.join(final_output_dir, 'best.pt'))
        
        # save pth
        torch.save({
            'epoch': epoch+1,
            'best_mIoU': best_mIoU,
            'best_epoch': best_epoch,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(final_output_dir,'checkpoint.pth.tar'))
        msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(valid_loss, mean_IoU, best_mIoU)
        
        logging.info(msg)
    
        # class 이름도 함께 출력하기
        if train_dataset.class_index_dict:
            class_index_dict = train_dataset.class_index_dict

            for index, IoU_value in enumerate(IoU_array):
                # index 0번은 ignore, 배경이기 때문에 index 1번부터 get
                class_name = class_index_dict.get(index, f"Unknown class index: {index}")
                logging.info(f"{class_name}: {IoU_value}")
            # logging.info(IoU_array)
        else:
            logging.info(IoU_array)

    # save final state
    torch.save(model.module.state_dict(),
            os.path.join(final_output_dir, 'final_state.pt'))

    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % np.int32((end-start)/3600))
    logger.info('Done')

if __name__ == '__main__':
    main()
