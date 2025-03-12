import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

sys.path.append('.') 
from sam2_train.sam2_image_predictor import SAM2ImagePredictor
from sam2_train.build_sam import build_sam2
from sam2_train.utils.transforms import SAM2Transforms

# Set device
device = "cpu"  # Force CPU mode
torch.backends.cudnn.enabled = False  # Disable cuDNN (CUDA Deep Learning Library)

# 1. Build the SAM Model
print("Device being used:", device)
sam_model = build_sam2(config_file="sam2_hiera_s.yaml", ckpt_path="./checkpoints/sam2_hiera_small.pt").to(device)
sam_model.eval()

# 2. Create the Image Predictor
predictor = SAM2ImagePredictor(sam_model)

# 3. Load and Prepare the Image
image_path = "brain_mri.jpg"  # Replace with your image path

# Precise type conversion function
def precise_image_conversion(image):
    """
    Carefully convert image to the exact type expected by the predictor
    """
    # If it's a PIL Image, keep it as is
    if isinstance(image, Image.Image):
        return np.array(image, dtype=np.uint8).copy()

    
    # If it's a numpy array, ensure it's the correct subtype
    if isinstance(image, np.ndarray):
        return image.astype(np.uint8)
    
    raise ValueError(f"Unsupported image type: {type(image)}")

# Load image
image = Image.open(image_path).convert("RGB")

plt.imshow(image)

# Convert image carefully
image_converted = precise_image_conversion(image)

# 4. Set Image for the Predictor
predictor.set_image(image_converted)

# 5. Define Prompts (Example: Box Prompt)
# The box prompt is expected in XYXY format: [x_min, y_min, x_max, y_max]
input_box = np.array([205, 403, 381, 563])  # This defines the box corners as described

# 6. Predict the Mask using the box prompt
masks, scores, logits = predictor.predict(
    box=input_box,           # Pass the box prompt here
    multimask_output=False,  # Set to True to get multiple masks if desired
    normalize_coords=True    # Normalizes the coordinates as expected by SAM
)

# 7. Display/Process the Mask
mask = masks[0] > 0  # Get the binary mask - shape (H, W), boolean

# Overlay the mask on the original image. Convert the image to float first to avoid clipping.
masked_image = image_converted.astype(float).copy()
masked_image[mask] = [0, 255, 0]  # Color the segmented region green
masked_image = np.clip(masked_image, 0, 255).astype(np.uint8)  # Clip to 0-255 and convert back

# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(masked_image)
plt.axis('off')
plt.title('Segmentation Result')
plt.show()

# Print additional information about the results
print("Mask shape:", mask.shape)
print("Mask dtype:", mask.dtype)
print("Scores:", scores)