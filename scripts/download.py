"""Download RCAEval cases from Hugging Face into data/raw.

Examples:
    python scripts/download.py                     # all of RE2 (main dataset)
    python scripts/download.py --suite re2 --system ob
    python scripts/download.py --suite re1
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "phamquiluan/RCAEval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="re2", choices=["re1", "re2", "re3", "all"])
    ap.add_argument("--system", default="", choices=["", "ob", "ss", "tt"])
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()

    prefix = "re" if args.suite == "all" else args.suite
    patterns = [f"{prefix}{args.system}*", "cases.parquet"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {patterns} from {REPO} into {out} ...")
    snapshot_download(repo_id=REPO, repo_type="dataset", allow_patterns=patterns, local_dir=str(out))

    cases = [p for p in out.iterdir() if p.is_dir() and p.name.startswith(prefix + args.system)]
    size = sum(f.stat().st_size for c in cases for f in c.rglob("*") if f.is_file())
    print(f"Done: {len(cases)} case folders, {size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
