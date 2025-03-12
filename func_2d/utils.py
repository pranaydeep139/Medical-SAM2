# func_2d/utils.py

""" helper function

author junde
"""

import logging
import math
import os
import pathlib
import random
import shutil
import sys
import time
import warnings
from collections import OrderedDict
from datetime import datetime
from typing import BinaryIO, List, Optional, Text, Tuple, Union

import dateutil.tz
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.utils as vutils
from PIL import Image
from torch.autograd import Function

import cfg

import pandas as pd

args = cfg.parse_args()
device = torch.device('cpu')  # Changed to CPU

def get_network(args, net, use_gpu=True, gpu_device=0, distribution=True):
    """Return the given network, using CPU."""
    if net == 'sam2':
        from sam2_train.build_sam import build_sam2
        net = build_sam2(args.sam_config, args.sam_ckpt, device="cpu")
        # Check that the returned model is a torch model (has a parameters method)
        if not hasattr(net, "parameters"):
            raise ValueError("The built network does not have a parameters() method. "
                             "Ensure that your configuration file is correct and that the model is instantiated properly.")
    else:
        print('The network name you have entered is not supported yet.')
        sys.exit()

    # if use_gpu:
    #     # Instead of sending to a GPU device, we send to CPU
    #     if distribution != 'none':
    #         net = torch.nn.DataParallel(net, device_ids=[int(id) for id in args.distributed.split(',')])
    #         net = net.to(device=device)
    #     else:
    #         net = net.to(device=device)

    return net


@torch.no_grad()
def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    **kwargs
) -> torch.Tensor:
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clamp(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding).narrow(
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[Text, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs
) -> None:
    """
    Save a given Tensor into an image file.
    """
    grid = make_grid(tensor, **kwargs)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)


def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))


def iou(outputs: np.array, labels: np.array): # unchanged for now, assumes binary per class
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou.mean()


class DiceCoeff(Function): # unchanged for now, assumes binary per class
    """Dice coeff for individual examples"""
    @staticmethod
    def forward(ctx, input, target):
        ctx.save_for_backward(input, target)
        eps = 0.0001
        ctx.inter = torch.dot(input.view(-1), target.view(-1))
        ctx.union = torch.sum(input) + torch.sum(target) + eps
        t = (2 * ctx.inter.float() + eps) / ctx.union.float()
        return t

    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors
        grad_input = grad_target = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * ctx.union - ctx.inter) / (ctx.union * ctx.union)
        if ctx.needs_input_grad[1]:
            grad_target = None
        return grad_input, grad_target


def dice_coeff(input, target): # unchanged for now, assumes binary per class
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device=input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()
    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff.apply(c[0], c[1]) # Use DiceCoeff.apply
    return s / (i + 1)


def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image


def view(tensor):
    image = tensor_to_img_array(tensor)
    assert len(image.shape) in [3, 4], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    Image.fromarray(image).show()


def vis_image(imgs, pred_masks, gt_masks, save_path, reverse=False, points=None):
    b, c, h, w = gt_masks.size() # Use gt_masks to get the number of classes 'c'
    dev = pred_masks.get_device()
    row_num = min(b, 4)
    if torch.max(pred_masks) > 1 or torch.min(pred_masks) < 0:
        pred_masks = torch.sigmoid(pred_masks)
    if reverse:
        pred_masks = 1 - pred_masks
        gt_masks = 1 - gt_masks


    if c > 1:  # for multi-class segmentation >= 2 classes (including background)
        preds_to_vis = []
        gts_to_vis = []
        # Assuming background is class 0, and other classes are 1, 2, ...
        # Visualize background (class 0), and the first foreground class (class 1)
        # You can extend this for more classes if needed, or visualize only foreground classes
        for i in range(min(c, 2)): # Visualize at most background and first foreground class
            pred_class_mask = pred_masks[:, i, :, :].unsqueeze(1).expand(b, 3, h, w) # Binary pred for class i
            gt_class_mask = gt_masks[:, i, :, :].unsqueeze(1).expand(b, 3, h, w)     # Binary gt for class i
            preds_to_vis.append(pred_class_mask)
            gts_to_vis.append(gt_class_mask)

        tup = [imgs[:row_num, :, :, :]] + preds_to_vis + gts_to_vis
        compose = torch.cat(tup, 0)
        vutils.save_image(compose, fp=save_path, nrow=row_num, padding=10)
    else: # if c == 1: # Original binary case - should still work if your one-hot encoding includes only one foreground class + background
        imgs = torchvision.transforms.Resize((h, w))(imgs)
        if imgs.size(1) == 1:
            imgs = imgs[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)
        pred_masks = pred_masks[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)
        gt_masks = gt_masks[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)
        if points is not None:
            for i in range(b):
                p = np.round(points.cpu() / args.image_size * args.out_size).to(dtype=torch.int)
                gt_masks[i, 0, p[i, 0]-2:p[i, 0]+2, p[i, 1]-2:p[i, 1]+2] = 0.5
                gt_masks[i, 1, p[i, 0]-2:p[i, 0]+2, p[i, 1]-2:p[i, 1]+2] = 0.1
                gt_masks[i, 2, p[i, 0]-2:p[i, 0]+2, p[i, 1]-2:p[i, 1]+2] = 0.4
        tup = (imgs[:row_num, :, :, :], pred_masks[:row_num, :, :, :], gt_masks[:row_num, :, :, :])
        compose = torch.cat(tup, 0)
        vutils.save_image(compose, fp=save_path, nrow=row_num, padding=10)
    return


def eval_seg(pred, true_mask_p, threshold):
    '''
    threshold: a int or a tuple of int
    masks: [b, num_classes, h, w] (one-hot encoded)
    pred: [b, 1, h, w] (binary prediction from model)
    '''
    b, num_classes, h, w = true_mask_p.size() # Get num_classes from true_mask_p

    if num_classes > 1: # Multi-class evaluation
        iou_scores = [0] * num_classes
        dice_scores = [0] * num_classes
        for class_index in range(num_classes): # Iterate through each class
            current_iou, current_dice = 0, 0
            for th in threshold:
                # Get binary ground truth mask for the current class
                gt_mask_binary = (true_mask_p[:, class_index, :, :] > th).float() # shape [b, h, w]
                # vpred is already binary from function.py (pred_binary)
                vpred_cpu = pred.cpu() # shape [b, 1, h, w]

                # Extract prediction for the current class (assuming binary prediction is for foreground, adapt if needed)
                pred_mask_binary = vpred_cpu[:, 0, :, :].numpy().astype('int32') # shape [b, h, w]
                gt_mask_binary_np = gt_mask_binary.squeeze(1).cpu().numpy().astype('int32') # shape [b, h, w]


                current_iou += iou(pred_mask_binary, gt_mask_binary_np) # Calculate IOU for current class
                current_dice += dice_coeff(pred[:, 0, :, :], gt_mask_binary).item() # Calculate Dice for current class

            iou_scores[class_index] = current_iou / len(threshold) # Average IOU over thresholds
            dice_scores[class_index] = current_dice / len(threshold) # Average Dice over thresholds

        return tuple(iou_scores), tuple(dice_scores) # Return tuples of IOU and Dice scores per class

    else: # Original binary evaluation (if num_classes is 1, though one-hot should have >= 2 classes)
        eiou, edice = 0, 0
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:, 0, :, :].numpy().astype('int32')
            disc_mask = gt_vmask_p[:, 0, :, :].squeeze(1).cpu().numpy().astype('int32')
            eiou += iou(disc_pred, disc_mask)
            edice += dice_coeff(vpred[:, 0, :, :], gt_vmask_p[:, 0, :, :]).item()

        return eiou / len(threshold), edice / len(threshold)



def random_click(mask):
    """
    Improved Version 1: Selects a random pixel from a randomly chosen *present* segment
    in a one-hot encoded mask (25 segments, no background). Ensures a click is generated
    if any segment is present in the mask.

    Args:
        mask (torch.Tensor or np.array): One-hot encoded mask of shape [num_classes, H, W], num_classes=25.

    Returns:
        tuple: (point_label, (row, col)) - The class index (0 to 24) and the coordinates of the random click.
               Returns (0, None) if no segment is found in the entire mask (highly unlikely for valid masks).
    """
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask

    num_classes, h, w = mask_np.shape

    present_segment_indices = [] # List to store indices of segments present in the mask
    for segment_index in range(num_classes):
        if np.any(mask_np[segment_index, :, :] > 0.5): # Check if any pixel for this segment is > 0.5
            present_segment_indices.append(segment_index)

    if not present_segment_indices: # If no segments are present in the mask at all
        return 0, None # Return no click

    # Now, choose a random segment index from the *present* segments only
    chosen_segment_index = np.random.choice(present_segment_indices)

    segment_indices = np.argwhere(mask_np[chosen_segment_index, :, :] > 0.5)
    random_index = segment_indices[np.random.randint(len(segment_indices))]
    row, col = random_index

    return chosen_segment_index, (row, col)


def agree_click(mask, label=1): # label is now class index
    """
    Selects a random pixel that agrees with a given class label in a one-hot encoded mask.
    If no pixel is found for the given label, tries to find one for class 0 (background).

    Args:
        mask (np.array): One-hot encoded mask of shape [num_classes, H, W].
        label (int, optional): The preferred class index to find a click for (default: 1).

    Returns:
        tuple: (label, (row, col)) - The class index and the coordinates of the click.
               Returns (opposite_label, (row, col)) if a click is found for the opposite label,
               or (0, None) if no pixel is found for either label.
    """
    # Ensure mask is a NumPy array
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask

    num_classes, h, w = mask_np.shape

    # Try to find indices for the given label (class index)
    label_indices = np.argwhere(mask_np[label, :, :] > 0.5) # Threshold, adjust if needed

    if len(label_indices) > 0:
        random_index = label_indices[np.random.randint(len(label_indices))]
        row, col = random_index
        return label, (row, col) # Found pixel for the given label

    else:
        # If no pixel found for the given label, try to find one for class 0 (background)
        opposite_label = 0 # Assuming class 0 is background
        opposite_indices = np.argwhere(mask_np[opposite_label, :, :] > 0.5) # Threshold, adjust if needed

        if len(opposite_indices) > 0:
            random_index = opposite_indices[np.random.randint(len(opposite_indices))]
            row, col = random_index
            return opposite_label, (row, col) # Found pixel for opposite label (background)
        else:
            return 0, None # No pixel found for either label, return background label and None


def random_box(multi_rater):
    max_value = torch.max(multi_rater[:, 0, :, :], dim=0)[0]
    max_value_position = torch.nonzero(max_value)
    x_coords = max_value_position[:, 0]
    y_coords = max_value_position[:, 1]
    x_min = int(torch.min(x_coords))
    x_max = int(torch.max(x_coords))
    y_min = int(torch.min(y_coords))
    y_max = int(torch.max(y_coords))
    x_min = random.choice(np.arange(x_min-10, x_min+11))
    x_max = random.choice(np.arange(x_max-10, x_max+11))
    y_min = random.choice(np.arange(y_min-10, y_min+11))
    y_max = random.choice(np.arange(y_max-10, y_max+11))
    return x_min, x_max, y_min, y_max