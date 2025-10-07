import cv2
import numpy as np
from .config import Config
from .utils.logger import timed

cfg = Config()

def _resize_max_w(img, max_w):
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / float(w)
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

@timed("preprocess")
def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = _resize_max_w(img, cfg.max_width_for_geometry)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    # geometry image (binarized)
    geo = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )

    # ---- mild deskew (only if we have enough foreground) ----
    coords = np.column_stack(np.where(geo > 0))
    if len(coords) > 10:
        rect = cv2.minAreaRect(coords.astype(np.float32))
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) > 0.1:  # avoid pointless ~0° rotations
            (h, w) = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img  = cv2.warpAffine(img,  M, (w, h), flags=cv2.INTER_CUBIC,  borderMode=cv2.BORDER_REPLICATE)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,  borderMode=cv2.BORDER_REPLICATE)
            geo  = cv2.warpAffine(geo,  M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

    # OCR image (clean, not binarized)
    ocr_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ocr_img = cv2.bilateralFilter(ocr_img, d=5, sigmaColor=30, sigmaSpace=30)

    return geo, ocr_img
