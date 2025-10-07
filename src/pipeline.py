from typing import List
from .preprocessor import preprocess
from .detector import detect_words
from .segmenter import segment_structure
from .assigner import assign_cells
from .ocr_engine import ocr_cells
from .postprocessor import to_matrix
from .exporter import export_csv
from .utils.logger import timed

@timed("pipeline")
def run_pipeline(img_path: str, out_dir: str, langs: List[str]):
    geo, ocr_img = preprocess(img_path)
    words = detect_words(ocr_img, langs)

    # Heuristic reading direction
    arabic = sum(sum("\u0600" <= ch <= "\u06FF" for ch in w["text"]) for w in words)
    latin  = sum(sum("A" <= ch <= "z" for ch in w["text"]) for w in words)
    rtl = arabic > latin

    grid = segment_structure(words, rtl=rtl)
    cells = assign_cells(words, grid)
    cells = ocr_cells(ocr_img, cells, langs)
    matrix = to_matrix(cells)
    export_csv(matrix, out_dir, img_path)
