import cv2
import numpy as np


def letterbox(
    image: np.ndarray,
    new_shape: int | tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scaleup: bool = True,
    stride: int = 32
) -> tuple[np.ndarray, float, tuple[float, float]]:

    """

    Resize and pad an image to a fixed shape while preserving aspect ratio.

    This function prepares real-world images for models that expect
    static input dimensions (for example, YOLO-based architectures).
    Instead of distorting the image, it resizes it proportionally and
    applies padding to reach the target shape.
    

    Parameters
    ----------
    image : np.ndarray
        Input image in HWC format.

    new_shape : int | tuple[int, int], optional
        Target (height, width) of the padded image. Defaults to 640x640.

    color : tuple[int, int, int], optional
        Padding color. Defaults to YOLO standard gray.

    auto : bool, optional
        If True, adjusts padding to be a multiple of `stride`.

    scaleup : bool, optional
        If False, prevents upscaling smaller images.

    stride : int, optional
        Model stride, typically 32 for YOLO architectures.

        
    Returns
    -------
    tuple[np.ndarray, float, tuple[float, float]]
        image : np.ndarray
            The padded image ready for inference.

        ratio : float
            Scaling ratio applied to the original image.

        pad : tuple[float, float]
            Horizontal and vertical padding applied (dw, dh).

    """

    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy array. Received: {image} with type {type(image)}")
    if isinstance(new_shape, int):
        new_shape: tuple[int,int] = (new_shape, new_shape)
    elif not (isinstance(new_shape, tuple) and len(new_shape) == 2):
        raise TypeError(f"new_shape must be an int or tuple of two ints. Received: {new_shape} with type {type(new_shape)}")
    if not (isinstance(color, tuple) and len(color) == 3):
        raise TypeError(f"color must be a tuple of 3 ints. Received: {color} with type {type(color)}")
    if not isinstance(auto, bool):
        raise TypeError(f"auto must be a boolean. Received: {auto} with type {type(auto)}")
    if not isinstance(scaleup, bool):
        raise TypeError(f"scaleup must be a boolean. Received: {scaleup} with type {type(scaleup)}")
    if not isinstance(stride, int):
        raise TypeError(f"stride must be an int. Received: {stride} with type {type(stride)}")


    shape: tuple[int,int] = image.shape[:2]

    ratio: float = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    if not scaleup:
        ratio = min(ratio, 1.0)

    new_unpadded: tuple[int,int] = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

    dw: float = new_shape[1] - new_unpadded[0]
    dh: float = new_shape[0] - new_unpadded[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpadded:
        image = cv2.resize(image, new_unpadded, interpolation=cv2.INTER_LINEAR)

    top: int = int(round(dh - 0.1))
    bottom: int = int(round(dh + 0.1))
    left: int = int(round(dw - 0.1))
    right: int = int(round(dw + 0.1))

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


    return image, ratio, (dw, dh)


def scale_coords(
    boxes: np.ndarray,
    original_shape: tuple[int,int],
    padded_shape: tuple[int,int],
    pad: tuple[float,float],
    ratio: float
) -> np.ndarray:

    """

    Scale bounding box coordinates back to the original image space.

    Model outputs are produced relative to the padded, resized input.
    This function reverses that transformation by removing padding and
    scaling coordinates back to the original resolution.


    Parameters
    ----------
    boxes : np.ndarray
        Bounding boxes in (x1, y1, x2, y2) format.

    original_shape : tuple[int, int]
        Shape of the original image (height, width).

    padded_shape : tuple[int, int]
        Shape of the padded input image.

    pad : tuple[float, float]
        Padding applied during letterboxing (dw, dh).

    ratio : float
        Scaling ratio used during resizing.


    Returns
    -------
    np.ndarray
        Scaled bounding boxes in original image coordinates.

    """

    if not isinstance(boxes, np.ndarray):
        raise TypeError(f"boxes must be a numpy array. Received: {boxes} with type {type(boxes)}")
    if not (isinstance(original_shape, tuple) and len(original_shape) == 2):
        raise TypeError(f"original_shape must be a tuple of two ints. Received: {original_shape} with type {type(original_shape)}")
    if not (isinstance(padded_shape, tuple) and len(padded_shape) == 2):
        raise TypeError(f"padded_shape must be a tuple of two ints. Received: {padded_shape} with type {type(padded_shape)}")
    if not (isinstance(pad, tuple) and len(pad) == 2):
        raise TypeError(f"pad must be a tuple of two floats. Received: {pad} with type {type(pad)}")
    if not isinstance(ratio, (int, float)):
        raise TypeError(f"ratio must be a float. Received: {ratio} with type {type(ratio)}")


    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]

    boxes[:, :4] /= ratio

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, original_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, original_shape[0])


    return boxes


def scale_masks(
    raw_masks: np.ndarray,
    padded_shape: tuple[int,int],
    original_shape: tuple[int,int],
    pad: tuple[float,float]
) -> np.ndarray:

    """

    Resize and crop predicted masks back to the original image size.

    Segmentation masks are first produced at a low resolution and
    aligned with the padded input. This helper removes padding,
    rescales masks, and aligns them with the original image.


    Parameters
    ----------
    raw_masks : np.ndarray
        Raw sigmoid-activated masks from the model.

    padded_shape : tuple[int, int]
        Shape of the padded input image.

    original_shape : tuple[int, int]
        Shape of the original image.

    pad : tuple[float, float]
        Padding applied during letterboxing (dw, dh).


    Returns
    -------
    np.ndarray
        Rescaled masks aligned with the original image.

    """

    if not isinstance(raw_masks, np.ndarray):
        raise TypeError(f"raw_masks must be a numpy array. Received: {raw_masks} with type {type(raw_masks)}")
    if not (isinstance(padded_shape, tuple) and len(padded_shape) == 2):
        raise TypeError(f"padded_shape must be a tuple of two ints. Received: {padded_shape} with type {type(padded_shape)}")
    if not (isinstance(original_shape, tuple) and len(original_shape) == 2):
        raise TypeError(f"original_shape must be a tuple of two ints. Received: {original_shape} with type {type(original_shape)}")
    if not (isinstance(pad, tuple) and len(pad) == 2):
        raise TypeError(f"pad must be a tuple of two floats. Received: {pad} with type {type(pad)}")


    h: int
    w: int
    h, w = padded_shape

    masks: np.ndarray = np.zeros((raw_masks.shape[0], original_shape[0], original_shape[1]))

    top: int = int(pad[1])
    left: int = int(pad[0])
    bottom: int = int(h - pad[1])
    right: int = int(w - pad[0])

    for i, mask in enumerate(raw_masks):
        mask_resized_to_padded: np.ndarray = cv2.resize(mask, (w, h))
        mask_cropped: np.ndarray = mask_resized_to_padded[top:bottom, left:right]
        masks[i] = cv2.resize(mask_cropped, (original_shape[1], original_shape[0]))


    return masks


def process_outputs(
    outputs: tuple[np.ndarray, np.ndarray],
    original_shape: tuple[int,int],
    padded_shape: tuple[int,int],
    pad: tuple[float,float],
    ratio: float,
    conf_threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """

    Convert raw model outputs into final boxes, classes, and masks.

    This function ties together all post-processing steps:
    filtering detections, scaling bounding boxes, reconstructing
    segmentation masks, and mapping everything back to the original
    image space.

    It assumes an exported YOLO segmentation model with nms=True.


    Parameters
    ----------
    outputs : tuple[np.ndarray, np.ndarray]
        Model outputs (detections, mask prototypes).

    original_shape : tuple[int, int]
        Shape of the original input image.

    padded_shape : tuple[int, int]
        Shape of the padded model input.

    pad : tuple[float, float]
        Padding applied during preprocessing.

    ratio : float
        Resize ratio applied during preprocessing.

    conf_threshold : float, optional
        Confidence threshold for filtering detections.


    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        boxes : np.ndarray
            Final bounding boxes in original image coordinates.

        class_ids : np.ndarray
            Predicted class indices.

        masks : np.ndarray
            Final segmentation masks.

    """

    if not (isinstance(outputs, (list, tuple)) and len(outputs) == 2):
        raise TypeError(f"outputs must be a tuple/list of length 2. Received: {outputs} with type {type(outputs)}")
    if not isinstance(conf_threshold, (int, float)):
        raise TypeError(f"conf_threshold must be a float. Received: {conf_threshold} with type {type(conf_threshold)}")
    if not (isinstance(original_shape, tuple) and len(original_shape) == 2):
        raise TypeError(f"original_shape must be a tuple of two ints. Received: {original_shape} with type {type(original_shape)}")
    if not (isinstance(padded_shape, tuple) and len(padded_shape) == 2):
        raise TypeError(f"padded_shape must be a tuple of two ints. Received: {padded_shape} with type {type(padded_shape)}")
    if not (isinstance(pad, tuple) and len(pad) == 2):
        raise TypeError(f"pad must be a tuple of two floats. Received: {pad} with type {type(pad)}")
    if not isinstance(ratio, (int, float)):
        raise TypeError(f"ratio must be a float. Received: {ratio} with type {type(ratio)}")


    detections: np.ndarray = outputs[0][0]
    prototypes: np.ndarray = outputs[1][0]

    valid_detections: np.ndarray = detections[detections[:, 4] > conf_threshold]

    if len(valid_detections) == 0:
        return np.array([]), np.array([]), np.array([])

    boxes: np.ndarray = valid_detections[:, :4]
    class_ids: np.ndarray = valid_detections[:, 5].astype(int)
    mask_coeffs: np.ndarray = valid_detections[:, 6:]

    scaled_boxes: np.ndarray = scale_coords(boxes, original_shape, padded_shape, pad, ratio)

    final_masks_raw: np.ndarray = mask_coeffs @ prototypes.reshape(32, -1)
    final_masks: np.ndarray = (1 / (1 + np.exp(-final_masks_raw))).reshape(-1, 160, 160)
    scaled_masks: np.ndarray = scale_masks(final_masks, padded_shape, original_shape, pad)


    return scaled_boxes, class_ids, scaled_masks
