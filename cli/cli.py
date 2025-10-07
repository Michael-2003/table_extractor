import argparse
from pathlib import Path
from src.pipeline import run_pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="data/outputs/")
    ap.add_argument("--langs", default="ar,en")
    args = ap.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    run_pipeline(args.input, args.output, langs)

if __name__ == "__main__":
    main()
