import cv2
import numpy as np
import os
from PIL import Image
import torch
import torch.nn.functional as F

###############################
# Part 1: Rasterize YOLO annotations to grayscale masks
###############################

def rasterize_yolo_to_mask(image_path, yolo_label_file, output_mask_path, image_size_hw):
    """
    Converts YOLO polygon annotations (normalized coordinates in text file)
    to a raster mask image (integer labels). In case of overlap, the first drawn
    segment is preserved.
    
    Args:
        image_path (str): Path to the image file (to get dimensions).
        yolo_label_file (str): Path to the YOLO label text file.
        output_mask_path (str): Path to save the output raster mask image.
        image_size_hw (tuple): Tuple (height, width) of the original image size.
    """
    height, width = image_size_hw
    # Use uint16 to support labels > 255
    mask = np.zeros((height, width), dtype=np.uint16)  

    with open(yolo_label_file, 'r') as f:
        lines = f.readlines()
        # Process each line in order
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                segment_id = int(parts[0])
            except ValueError:
                continue
            polygon_coords = parts[1:]
            if len(polygon_coords) < 6:
                continue  # Need at least 3 points
            points_pixel = []
            for i in range(0, len(polygon_coords), 2):
                try:
                    x_norm = float(polygon_coords[i])
                    y_norm = float(polygon_coords[i+1])
                except ValueError:
                    continue
                x_pixel = int(round(x_norm * width))
                y_pixel = int(round(y_norm * height))
                points_pixel.append([x_pixel, y_pixel])
            if len(points_pixel) < 3:
                continue
            contour = np.array(points_pixel, np.int32).reshape((-1, 1, 2))
            
            # Create a temporary mask for this polygon
            temp = np.zeros_like(mask)
            cv2.fillPoly(temp, [contour], color=segment_id)
            # Only update pixels that haven't been set yet
            mask[(mask == 0) & (temp != 0)] = segment_id

    # Determine data type and mode for saving based on maximum label value
    max_val = mask.max()
    if max_val < 256:
        # Safe to cast to 8-bit for visualization
        mask = mask.astype(np.uint8)
        mode = 'L'
    else:
        mode = 'I;16'
    
    mask_image = Image.fromarray(mask, mode=mode)
    mask_image.save(output_mask_path)
    print(f"Saved grayscale mask to {output_mask_path} (max label: {max_val}, mode: {mode})")


def process_images(image_dir, label_dir, output_mask_dir, image_size_hw):
    os.makedirs(output_mask_dir, exist_ok=True)
    for image_filename in os.listdir(image_dir):
        if image_filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_name_no_ext = os.path.splitext(image_filename)[0]
            image_path = os.path.join(image_dir, image_filename)
            yolo_label_file = os.path.join(label_dir, image_name_no_ext + '.txt')
            output_mask_file = os.path.join(output_mask_dir, image_name_no_ext + '_mask.png')
            
            if os.path.exists(yolo_label_file):
                rasterize_yolo_to_mask(image_path, yolo_label_file, output_mask_file, image_size_hw)
            else:
                print(f"Warning: No label file found for image: {image_filename}")


###############################
# Part 2: Convert the grayscale masks to one-hot encoding
###############################

def mask_to_one_hot(mask_path, num_classes):
    """
    Converts a segmentation mask to one-hot encoding.

    Args:
        mask_path (str): Path to the mask image.
        num_classes (int): Total number of classes (including background).

    Returns:
        torch.Tensor: One-hot encoded tensor of shape (num_classes, height, width).
    """
    # Load the mask image (assumed to be single-channel)
    mask_image = Image.open(mask_path).convert("L")
    mask_array = np.array(mask_image)
    mask_tensor = torch.tensor(mask_array, dtype=torch.int64)
    one_hot_tensor = F.one_hot(mask_tensor, num_classes=num_classes)
    one_hot_tensor = one_hot_tensor.permute(2, 0, 1).float()
    return one_hot_tensor


def process_one_hot_masks(raster_mask_dir, one_hot_dir, num_classes=None):
    """
    Processes all raster mask images in raster_mask_dir, converts each mask
    to one-hot encoding, and saves the one-hot tensors as .pt files in one_hot_dir.
    
    Args:
        raster_mask_dir (str): Directory containing raster mask PNGs.
        one_hot_dir (str): Output directory for one-hot encoded masks.
        num_classes (int, optional): Total number of classes. If not provided,
                                     the function computes it as max(mask)+1.
    """
    os.makedirs(one_hot_dir, exist_ok=True)
    for filename in os.listdir(raster_mask_dir):
        if filename.lower().endswith('.png'):
            mask_path = os.path.join(raster_mask_dir, filename)
            # If num_classes is not provided, compute it from the mask:
            if num_classes is None:
                mask_array = np.array(Image.open(mask_path).convert("L"))
                num_classes_used = int(mask_array.max()) + 1
            else:
                num_classes_used = num_classes
            one_hot = mask_to_one_hot(mask_path, num_classes_used)
            # Save the one-hot encoded tensor as a .pt file
            output_filename = os.path.splitext(filename)[0] + '_onehot.pt'
            output_path = os.path.join(one_hot_dir, output_filename)
            torch.save(one_hot, output_path)
            print(f"Saved one-hot mask to {output_path} (num_classes: {num_classes_used})")


###############################
# Main script: Define folder paths and run processing
###############################

# Define folder paths (adjust these to your folder structure)
# For training split:
train_image_dir = 'training/images/train'
train_label_dir = 'training/labels/train'
train_raster_mask_dir = 'training/raster_masks/train'
train_one_hot_dir = 'training/one_hot_masks/train'

# For validation split:
val_image_dir = 'training/images/val'
val_label_dir = 'training/labels/val'
val_raster_mask_dir = 'training/raster_masks/val'
val_one_hot_dir = 'training/one_hot_masks/val'

# Define the image size (height, width)
image_size_hw = (2000, 2000)  # Adjust as needed per your images

# Step 1: Convert YOLO annotations to raster masks
process_images(train_image_dir, train_label_dir, train_raster_mask_dir, image_size_hw)
process_images(val_image_dir, val_label_dir, val_raster_mask_dir, image_size_hw)
print("Raster mask conversion complete! Masks saved under training/raster_masks/")

# Step 2: Convert raster masks to one-hot encoding.
# If you know the number of classes, set num_classes; otherwise, leave as None to compute from each mask.
process_one_hot_masks(train_raster_mask_dir, train_one_hot_dir, num_classes=None)
process_one_hot_masks(val_raster_mask_dir, val_one_hot_dir, num_classes=None)
print("One-hot encoding conversion complete! One-hot masks saved under training/one_hot_masks/")
