from PIL import Image
import numpy as np
import os

def print_intensity_values(image_path):
    # Open the image and convert it to grayscale ('L' mode)
    img = Image.open(image_path).convert("L")
    # Convert the image to a numpy array
    arr = np.array(img)
    # Get the unique intensity values
    unique_intensities = np.unique(arr)
    print("Unique intensity values in the image:")
    print(unique_intensities)

# Example usage:
png_path = "training/raster_masks/train/141_274_mask.png"
print_intensity_values(png_path)

def get_label_ids(label_file_path):
    """
    Reads a label text file and returns a set of segment ids.
    Each line starts with an integer (segment id), followed by coordinates.
    """
    segment_ids = set()
    segment_ids.add(0)
    with open(label_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                segment_id = int(parts[0])
                segment_ids.add(segment_id)
            except ValueError:
                continue
    return segment_ids

def get_mask_ids(mask_file_path):
    """
    Opens a PNG mask image in grayscale and returns a set of unique intensity values.
    """
    img = Image.open(mask_file_path).convert("L")
    arr = np.array(img)
    unique_ids = {int(val) for val in np.unique(arr)}
    return unique_ids

def compare_labels_and_masks(labels_base_dir, raster_masks_base_dir):
    """
    Compares label text files and corresponding raster mask PNG files for both train and val splits.
    For a label file named 'xxx.txt', the corresponding raster mask file is expected to be 'xxx_mask.png'
    in the same split folder.
    
    Prints:
      - Total label files processed.
      - Total raster mask files processed.
      - Number of pairs with matching unique segment ids.
    """
    splits = ["train", "val"]
    total_labels = 0
    total_masks = 0
    matching_pairs = 0
    mismatches = []
    
    for split in splits:
        label_dir = os.path.join(labels_base_dir, split)
        mask_dir = os.path.join(raster_masks_base_dir, split)
        
        # List label files (assuming .txt)
        label_files = [f for f in os.listdir(label_dir) if f.lower().endswith('.txt')]
        total_labels += len(label_files)
        
        for label_file in label_files:
            base_name = os.path.splitext(label_file)[0]
            # Expected corresponding mask filename is base_name + '_mask.png'
            mask_file = base_name + '_mask.png'
            label_file_path = os.path.join(label_dir, label_file)
            mask_file_path = os.path.join(mask_dir, mask_file)
            
            if not os.path.exists(mask_file_path):
                print(f"Warning: No corresponding mask file for {label_file}")
                continue
            
            total_masks += 1
            
            label_ids = get_label_ids(label_file_path)
            mask_ids = get_mask_ids(mask_file_path)
            
            if label_ids == mask_ids:
                matching_pairs += 1
            else:
                mismatches.append((label_file, label_ids, mask_ids))
    
    print("Comparison Report:")
    print(f"Total label files processed: {total_labels}")
    print(f"Total raster mask files processed: {total_masks}")
    print(f"Number of matching pairs: {matching_pairs}")
    
    if mismatches:
        print("\nMismatched files:")
        for file, lab_ids, msk_ids in mismatches:
            print(f"  {file}:")
            print(f"    Label file segment ids: {sorted(lab_ids)}")
            print(f"    Mask file intensity values: {sorted(msk_ids)}")
    else:
        print("\nAll corresponding files have matching segment ids.")

# Example usage:
labels_base_dir = "training/labels"
raster_masks_base_dir = "training/raster_masks"

compare_labels_and_masks(labels_base_dir, raster_masks_base_dir)
