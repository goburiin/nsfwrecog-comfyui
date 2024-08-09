import torch
import numpy as np
from PIL import Image
from nsfwrecog.nsfwrecog.nsfwrecog import NsfwRecog
from collections import namedtuple

# Correct SEG definition
SEG = namedtuple("SEG", ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'], defaults=[None])

class NSFWDetectorNode:
    def __init__(self):
        self.detector = NsfwRecog()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "detect"

    def detect(self, image):
        print(f"Input image shape: {image.shape}")

        if isinstance(image, torch.Tensor):
            image_np = image.squeeze().cpu().numpy()
        else:
            image_np = np.array(image)

        if image_np.ndim == 3 and image_np.shape[0] in [1, 3, 4]:
            image_np = np.transpose(image_np, (1, 2, 0))
        if image_np.ndim == 2:
            image_np = np.stack((image_np,) * 3, axis=-1)

        if image_np.dtype != np.uint8:
            if np.issubdtype(image_np.dtype, np.floating):
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)

        print(f"Processed numpy array shape: {image_np.shape}")

        pil_image = Image.fromarray(image_np)
        detected_results = self.detector.detect(pil_image)
        print(f"Detected results: {detected_results}")

        items = []
        h, w = image_np.shape[:2]

        for result in detected_results:
            bbox = result['bbox']
            label = result['class']
            confidence = result.get('score', 1.0)

            x1, y1, x2, y2 = bbox
            crop_region = (x1, y1, x2, y2)

            mask = np.zeros((h, w), dtype=np.float32)
            mask[y1:y2, x1:x2] = 1.0

            cropped_mask = mask[y1:y2, x1:x2]

            seg = SEG(cropped_image=None, 
                      cropped_mask=cropped_mask, 
                      confidence=confidence, 
                      crop_region=crop_region, 
                      bbox=bbox, 
                      label=label, 
                      control_net_wrapper=None)
            items.append(seg)

        segs = ((h, w), items)
        print(f"Number of SEG items: {len(items)}")
        print(f"SEGS shape: {segs[0]}")

        return (segs,)

NODE_CLASS_MAPPINGS = {
    "NSFWDetectorNode": NSFWDetectorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NSFWDetectorNode": "NSFW Detector"
}
