# train_2d.py
#!/usr/bin/env python3

""" train network using pytorch
    Jiayuan Zhu
"""

import os
import time

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
#from dataset import *
from torch.utils.data import DataLoader

import cfg
import func_2d.function as function
from conf import settings
from func_2d.dataset.custom_dataset import CustomDataset
#from models.discriminatorlayer import discriminator
from func_2d.dataset import *
from func_2d.utils import *


def main():
    # use bfloat16 for the entire work (CPU version)
    torch.autocast(device_type="cpu", dtype=torch.bfloat16).__enter__()

    # (GPU-specific tf32 settings removed for CPU)

    args = cfg.parse_args()
    GPUdevice = torch.device('cpu')

    # **CHANGE HERE**: Set args.image_size to 1024 to match sam2_hiera_s config
    args.image_size = 1024
    print(f"DEBUG: args.image_size set to {args.image_size}") # Debug print

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)

    # optimisation
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    '''load pretrained model'''

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    '''segmentation data'''
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    # example of REFUGE dataset
    if args.dataset == 'custom_data':
        '''custom_data'''
        train_data_path = 'training/images/train'  # Path to your training images
        # **CHANGE HERE**: Point to the one-hot mask directory
        train_mask_path = 'training/one_hot_masks/train'
        test_data_path = 'training/images/val'  # Path to your validation images
        # **CHANGE HERE**: Point to the one-hot mask directory
        test_mask_path = 'training/one_hot_masks/val'

        limit_size = 8

        train_dataset = CustomDataset(args, train_data_path, train_mask_path, transform=transform_train, mode='Training', prompt=args.prompt, limit_dataset_size=limit_size)
        for i in range(len(train_dataset)):
            sample = train_dataset[i]
            if sample is None:
                print(f"Sample {i} is None")
            else:
                print(f"Sample {i} loaded successfully.")

        test_dataset = CustomDataset(args, test_data_path, test_mask_path, transform=transform_test, mode='Test', prompt=args.prompt, limit_dataset_size=limit_size)
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            if sample is None:
                print(f"Sample {i} is None")
            else:
                print(f"Sample {i} loaded successfully.")

        nice_train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=2, pin_memory=True)
        nice_test_loader = DataLoader(test_dataset, batch_size=args.b, shuffle=False, num_workers=2, pin_memory=True)
    '''end'''

    '''checkpoint path and tensorboard'''
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW))

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    '''begain training'''
    best_tol = 1e4
    best_dice = 0.0

    num_test_epochs = 2

    for epoch in range(num_test_epochs):

        if epoch == 0:
            tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
            logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

        net.train()
        time_start = time.time()
        loss = function.train_sam(args, net, optimizer, nice_train_loader, epoch, writer)
        logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        net.eval()
        if epoch % args.val_freq == 0 or epoch == num_test_epochs-1:
            tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
            logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

            if edice > best_dice:
                best_dice = edice
                torch.save({'model': net.state_dict(), 'parameter': net._parameters},
                           os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth'))

    writer.close()


if __name__ == '__main__':
    main()