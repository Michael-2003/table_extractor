from typing import List, Dict
import numpy as np
from .utils.logger import timed

try:
    import easyocr
except Exception:
    easyocr = None

_reader_cache = {}

@timed("detect_words")
def detect_words(img_ocr: np.ndarray, langs: List[str]) -> List[Dict]:
    if easyocr is None:
        raise ImportError("easyocr is not installed. Please `pip install easyocr`.")

    key = tuple(sorted(langs))
    if key not in _reader_cache:
        _reader_cache[key] = easyocr.Reader(langs, gpu=False, verbose=False)
    reader = _reader_cache[key]

    result = reader.readtext(img_ocr)
    out = []
    for (pts, text, conf) in result:
        if not text:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        out.append({'text': text.strip(), 'conf': float(conf), 'bbox': [x1, y1, x2, y2]})
    return out
