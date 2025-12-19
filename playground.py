import numpy as np

from deployment_helpers import letterbox, process_outputs

# Step 1: Create a dummy image (RGB, 480x640)
dummy_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


# Step 2: Preprocess the image using letterbox
padded_image, ratio, pad = letterbox(
    dummy_image,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleup=True,
    stride=32
)

print("Padded image shape:", padded_image.shape)
print("Resize ratio:", ratio)
print("Padding applied:", pad)


# Step 3: Create dummy model outputs
# detections shape: (1, num_detections, 4+1+1+32) = (1, 3, 38)
num_detections = 3
num_mask_coeffs = 32
detections = np.random.rand(1, num_detections, 38).astype(np.float32)

# prototypes shape: (1, 32, 160, 160)
prototypes = np.random.rand(1, num_mask_coeffs, 160, 160).astype(np.float32)

dummy_outputs = (detections, prototypes)


# Step 4: Process model outputs
boxes, class_ids, masks = process_outputs(
    outputs=dummy_outputs,
    original_shape=dummy_image.shape[:2],
    padded_shape=padded_image.shape[:2],
    pad=pad,
    ratio=ratio,
    conf_threshold=0.0  # keep all dummy detections
)

print("Boxes shape:", boxes.shape)
print("Class IDs shape:", class_ids.shape)
print("Masks shape:", masks.shape)


# Step 5: Print first detection details
if boxes.shape[0] > 0:
    for i, box in enumerate(boxes):
        print(f"Detection {i}: box={box}, class_id={class_ids[i]}")
