from pathlib import Path
from src.pipeline import run_pipeline
import argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Borderless Table Extractor (PoC, EasyOCR, CPU-only)")
    p.add_argument("--input", required=True, help="Path to image file")
    p.add_argument("--output", default="data/outputs/", help="Output directory")
    p.add_argument("--langs", default="ar,en", help="Comma-separated languages for OCR")
    args = p.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    run_pipeline(args.input, args.output, langs)
