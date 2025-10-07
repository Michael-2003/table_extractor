from typing import List
import numpy as np
import cv2
from .config import Config
from .utils.geometry import expand_bbox


try:
    import easyocr
except Exception:
    easyocr = None


cfg = Config()
_reader = None


def _get_reader(langs: List[str]):
    global _reader
    if _reader is None:
        if easyocr is None:
            raise ImportError("easyocr is not installed. Please `pip install easyocr`.")
        _reader = easyocr.Reader(langs, gpu=False, verbose=False)
    return _reader


def ocr_cells(img_ocr: np.ndarray, cells, langs: List[str]):
    h, w = img_ocr.shape[:2]
    rdr = _get_reader(langs)
    for row in cells:
        for c in row:
            # If we already have texts from detection stage, use them
            if c['texts']:
                joined = []
                confs = []
                for t in c['texts']:
                    if t['text']:
                        joined.append(t['text'])
                        confs.append(float(t['conf']))
                if joined:
                    c['value'] = "\n".join(joined)
                    c['conf'] = sum(confs)/max(1,len(confs))
                    continue
            # Otherwise, OCR the crop
            x1,y1,x2,y2 = expand_bbox(c['bbox'], cfg.cell_pad_px, w, h)
            crop = img_ocr[y1:y2, x1:x2]
            res = rdr.readtext(crop)
            vals = []
            confs = []
            for (pts,text,conf) in res:
                if not text:
                    continue
                vals.append(text.strip())
                confs.append(float(conf))
            c['value'] = "\n".join(vals) if vals else ""
            c['conf'] = sum(confs)/max(1,len(confs)) if confs else 0.0
    return cells