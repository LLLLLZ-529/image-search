import numpy as np
from PIL import Image

def center_crop(img_path, crop_size=224):
    # Step 1: load image
    image = Image.open(img_path).convert("RGB")

    # Step 2: center crop
    w, h = image.size
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    image = image.crop((left, top, right, bottom))  # PIL Image, size (224, 224)

    # Step 3: to_numpy
    image = np.array(image).astype(np.float32) / 255.0  # (H, W, C)

    # Step 4: norm
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std  # (H, W, C)
    image = image.transpose(2, 0, 1) # (C, H, W)
    return image[None] # (1, C, H, W)

def resize_short_side(img_path, target_size=224, patch_size=14):
    """Resize while keeping aspect ratio; short side == target_size and both dims divisible by patch_size."""
    image = Image.open(img_path).convert("RGB")

    w, h = image.size
    short, long = (w, h) if w < h else (h, w)
    if short == 0:
        raise ValueError("Invalid image size with zero dimension.")

    scale = target_size / short
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    image = image.resize((new_w, new_h), Image.BICUBIC)

    def _center_crop_to_multiple(img, divisor):
        cur_w, cur_h = img.size
        crop_w = (cur_w // divisor) * divisor
        crop_h = (cur_h // divisor) * divisor
        if crop_w == cur_w and crop_h == cur_h:
            return img
        left = (cur_w - crop_w) // 2
        top = (cur_h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h
        return img.crop((left, top, right, bottom))

    image = _center_crop_to_multiple(image, patch_size)

    image = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = image.transpose(2, 0, 1)
    return image[None]