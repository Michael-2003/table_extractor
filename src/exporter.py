from pathlib import Path
import csv
from .config import Config

cfg = Config()

def export_csv(matrix, out_dir: str, in_path: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(out_dir) / (Path(in_path).stem + ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, dialect=cfg.csv_dialect)
        for row in matrix:
            w.writerow(row)
    print(f"[export] CSV -> {csv_path}")
