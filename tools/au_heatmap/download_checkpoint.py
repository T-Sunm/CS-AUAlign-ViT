"""Download disfa_adaptation_f0.pth (the 12-AU DISFA heatmap model) and verify its sha256.

The same archive is linked from two repos by the same author; both links are tried.
"""
import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

URLS = [
    "https://www.dropbox.com/s/qzeugjqi4ss2w5m/au_detector.tar?dl=1",  # WarpedGANSpace / PDETraversal
    "https://www.dropbox.com/s/jkkf1gda9o8ed47/au_detector.tar?dl=1",  # ContraCLIP
]
SHA256 = "dbdf18bf541de3c46769d712866bef38496b7528072850c28207747b2b2c101e"
OUT_DIR = Path(__file__).resolve().parent / "checkpoints"
OUT_FILE = OUT_DIR / "disfa_adaptation_f0.pth"


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if OUT_FILE.exists():
        print(f"Already exists: {OUT_FILE}")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "au_detector.tar"
        for url in URLS:
            try:
                print(f"Downloading {url}")
                urllib.request.urlretrieve(url, tar_path)
            except Exception as e:
                print(f"  failed: {e}")
                continue
            digest = sha256sum(tar_path)
            if digest != SHA256:
                print(f"  sha256 mismatch ({digest}), trying next link")
                continue
            print("  sha256 OK")
            with tarfile.open(tar_path) as tar:
                member = next((m for m in tar.getmembers() if m.name.endswith("disfa_adaptation_f0.pth")), None)
                if member is None:
                    sys.exit("disfa_adaptation_f0.pth not found in archive: " + ", ".join(tar.getnames()))
                with tar.extractfile(member) as src, open(OUT_FILE, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            print(f"Saved {OUT_FILE}")
            return
    sys.exit("Could not download the checkpoint from any link. Open the links in a browser and put "
             "disfa_adaptation_f0.pth into ./checkpoints/ manually.")


if __name__ == "__main__":
    main()
