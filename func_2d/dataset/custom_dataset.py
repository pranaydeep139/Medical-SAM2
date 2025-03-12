import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np # Make sure to import numpy
import torch.nn.functional as F # Make sure to import F

# Assuming you've correctly pasted the improved random_click function into utils.py
from func_2d.utils import random_click # Import the IMPROVED version

class CustomDataset(Dataset):
    def __init__(self, args, data_path, mask_path, transform=None, mode='Training', prompt='click', limit_dataset_size = None):
        self.image_dir = data_path
        self.mask_dir = mask_path
        self.image_filenames = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        if limit_dataset_size is not None:
            self.image_filenames = self.image_filenames[:limit_dataset_size] # Take the first 'limit_dataset_size' filenames

        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.mask_size = args.out_size
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_filename = self.image_filenames[index]
        image_name_no_ext = os.path.splitext(image_filename)[0]

        # Load Image
        img_path = os.path.join(self.image_dir, image_filename)
        img = Image.open(img_path).convert('RGB')

        # Load One-Hot Encoded Mask
        mask_filename = image_name_no_ext + '_mask_onehot' + '.pt' # Corrected mask filename
        mask_path = os.path.join(self.mask_dir, mask_filename)
        mask_tensor = torch.load(mask_path)  # Shape: [num_classes, H, W]

        # Apply Image Transform
        if self.transform:
            img = self.transform(img)

        # Resize Mask if Necessary
        if mask_tensor.shape[1:] != (self.mask_size, self.mask_size):
            mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=(self.mask_size, self.mask_size), mode='nearest').squeeze(0)

        # Prompt Generation (if applicable)
        point_label, pt_normalized = None, None
        print(f"DEBUG: Before prompt block - index: {index}, prompt_mode: {self.prompt}") # DEBUG PRINT 1: Before prompt block

        if self.prompt == 'click':
            print(f"DEBUG: Inside click prompt block - index: {index}") # DEBUG PRINT 2: Inside click block
            mask_np = mask_tensor.cpu().numpy() # Convert mask_tensor to NumPy array for random_click function
            point_label_index, pt = random_click(mask_np) # Use IMPROVED Version 1

            if pt is not None:
                point_label = point_label_index # Use the segment index as point_label
                pt_normalized_list = [pt[1] / self.img_size, pt[0] / self.img_size] # Normalize to 0-1
                pt_normalized = torch.tensor(pt_normalized_list, dtype=torch.float32) # Convert list to tensor, float32 dtype
                print(f"DEBUG: point_label assigned: {point_label}, pt_normalized assigned: {pt_normalized}") # DEBUG PRINT 3: After assignment
            else:
                point_label = None # Or handle no click case as needed
                pt_normalized = None
                print(f"DEBUG: point_label is None, pt_normalized is None (no click generated)") # DEBUG PRINT 4: No click

        else:
            pt_normalized = None

        print(f"DEBUG: After prompt block - point_label: {point_label}, pt_normalized: {pt_normalized}") # DEBUG PRINT 5: After prompt block

        image_meta_dict = {'filename_or_obj': image_filename}
        sample = { # Create sample dictionary
            'image': img,
            'mask': mask_tensor,
            'p_label': point_label,
            'pt': pt_normalized,
            'image_meta_dict': image_meta_dict,
        }
        print(f"DEBUG: Sample dictionary created - keys: {sample.keys()}, p_label type: {type(sample['p_label'])}, pt type: {type(sample['pt'])}") # DEBUG PRINT 6: Sample dict
        return sample