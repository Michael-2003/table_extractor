from dataclasses import dataclass

@dataclass
class Config:
    max_width_for_geometry: int = 1600
    row_cluster_eps: int = 12     # Y clustering tolerance (px)
    min_col_gap_px: int = 20      # minimum valley to accept a column cut
    snap_max_shift_px: int = 12   # reserved (kept for future), safe to ignore
    cell_pad_px: int = 3          # crop padding for OCR
    ocr_min_conf: float = 0.40    # threshold if you ever want re-OCR logic
    csv_dialect: str = "excel"
