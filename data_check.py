import os
import torch

def main():
    # Set the directory for validation one-hot masks.
    # Adjust this path to match your folder structure.
    one_hot_val_dir = "training/one_hot_masks/val"

    all_unique_classes = set()

    # Iterate only over files ending with '_onehot.pt' in the validation folder.
    for filename in os.listdir(one_hot_val_dir):
        if filename.endswith("_onehot.pt"):
            file_path = os.path.join(one_hot_val_dir, filename)
            one_hot = torch.load(file_path)  # Expected shape: (num_classes, H, W)
            
            # Convert one-hot encoding to label mask by taking argmax over channel dimension.
            label_mask = torch.argmax(one_hot, dim=0)  # shape: (H, W)
            unique_classes = torch.unique(label_mask).cpu().numpy().tolist()
            print(f"{filename}: unique classes = {sorted(unique_classes)}")
            
            all_unique_classes.update(unique_classes)

    print("Union of unique classes in validation set:", sorted(all_unique_classes))

if __name__ == "__main__":
    main()
