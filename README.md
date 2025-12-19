# YOLO Deployment Helper Functions

<div style="display: flex; gap: 10px; flex-wrap: wrap;">
    <img src="https://img.shields.io/github/license/AliKHaliliT/YOLO-Deployment-Helper-Functions" alt="License">
    <img src="https://img.shields.io/github/last-commit/AliKHaliliT/YOLO-Deployment-Helper-Functions" alt="Last Commit">
    <img src="https://img.shields.io/github/issues/AliKHaliliT/YOLO-Deployment-Helper-Functions" alt="Open Issues">
</div>
<br/>

A set of helper functions for preprocessing, postprocessing, and deploying YOLO segmentation models. Provides utilities for handling ONNX or PyTorch model outputs, resizing inputs, and scaling masks and bounding boxes for real-world inference pipelines.

## Usage

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/AliKHaliliT/YOLO-Deployment-Helper-Functions.git
pip install numpy==2.0.2 opencv-python==4.10.0.84
```

### Example

This example demonstrates how to use the provided helper functions with dummy inputs:

```python
from deployment_helpers import letterbox, scale_coords, scale_masks, process_outputs
import numpy as np

# Create a dummy image
dummy_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

# Preprocess image
padded_image, ratio, pad = letterbox(dummy_image, new_shape=(640, 640))

# Dummy model outputs
num_detections = 3
num_mask_coeffs = 32
detections = np.random.rand(1, num_detections, 38).astype(np.float32)
prototypes = np.random.rand(1, num_mask_coeffs, 160, 160).astype(np.float32)
dummy_outputs = (detections, prototypes)

# Process outputs
boxes, class_ids, masks = process_outputs(
    outputs=dummy_outputs,
    original_shape=dummy_image.shape[:2],
    padded_shape=padded_image.shape[:2],
    pad=pad,
    ratio=ratio,
    conf_threshold=0.0
)

print("Boxes shape:", boxes.shape)
print("Class IDs shape:", class_ids.shape)
print("Masks shape:", masks.shape)
```

### Functions Overview

| Function          | Description                                                                                                                                       |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `letterbox`       | Resizes and pads an image to a target shape while preserving aspect ratio. Supports optional stride alignment.                                    |
| `scale_coords`    | Rescales bounding boxes from padded image coordinates back to the original image coordinates.                                                     |
| `scale_masks`     | Rescales predicted masks from the padded size back to the original image size.                                                                    |
| `process_outputs` | Converts raw YOLO model outputs (detections and mask prototypes) into final boxes, class IDs, and masks. Supports confidence threshold filtering. |

### Notes

- Designed for YOLO segmentation models exported to ONNX or PyTorch.
- Static input/output shapes improve inference performance and predictability.
- Letterboxing is required to maintain aspect ratio and prevent distortions in real-world images.
- Dummy inputs can be used to test the pipeline without requiring a trained model.

## License

This work is under an [MIT](https://choosealicense.com/licenses/mit/) License.
