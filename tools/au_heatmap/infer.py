"""Run the 12-AU heatmap model on images.

Examples
  python infer.py --input face.jpg
  python infer.py --input folder_of_images/ --out results/ --detector sfd
  python infer.py --input already_cropped_faces/ --detector none

Outputs per image (largest face only):
  <name>_heatmaps.npy   float32, 12 x 64 x 64 (channel order = AU 1,2,4,5,6,9,12,15,17,20,25,26)
  <name>_heatmaps.png   face crop + 12 heatmap overlays with predicted intensity
  intensities.csv       one row per image, one column per AU
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from au_heatmap import AU_IDS, AU_NAMES, AUHeatmapDetector

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------- face detection
class FaceDetector:
    def __init__(self, kind, device):
        self.kind = kind
        if kind == "sfd":
            # Same detector family as the original evaluation code. pip install face-alignment
            from face_alignment.detection.sfd.sfd_detector import SFDDetector
            self.det = SFDDetector(device=device)
        elif kind == "haar":
            self.det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def largest_box(self, rgb):
        h, w = rgb.shape[:2]
        if self.kind == "none":
            return 0, 0, w, h
        if self.kind == "sfd":
            boxes = [b[:4] for b in self.det.detect_from_image(rgb.copy()) if b[4] > 0.5]
        else:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            boxes = [(x, y, x + bw, y + bh) for x, y, bw, bh in self.det.detectMultiScale(gray, 1.1, 5)]
        if not boxes:
            return None
        return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def square_crop(rgb, box, margin):
    """Square crop around the box (with relative margin), padded with zeros if it leaves the image."""
    x1, y1, x2, y2 = map(float, box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + margin)
    x1, y1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    s = int(round(side))
    h, w = rgb.shape[:2]
    out = np.zeros((s, s, 3), dtype=rgb.dtype)
    sx1, sy1, sx2, sy2 = max(x1, 0), max(y1, 0), min(x1 + s, w), min(y1 + s, h)
    out[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = rgb[sy1:sy2, sx1:sx2]
    return cv2.resize(out, (256, 256), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------- visualisation
def save_overlay(face256, heatmaps, intensities, path):
    tiles = [cv2.putText(face256.copy(), "input", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)]
    vmax = max(5.0, float(heatmaps.max()))        # AU intensity scale is 0..5
    for i in range(12):
        hm = cv2.resize(np.clip(heatmaps[i], 0, None) / vmax, (256, 256))
        color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
        tile = cv2.addWeighted(face256, 0.5, color, 0.5, 0)
        label = f"AU{AU_IDS[i]}: {intensities[i]:.2f}"
        cv2.putText(tile, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(tile)
    tiles.append(np.zeros_like(face256))           # 13 tiles + 1 blank -> 2 rows x 7
    grid = np.vstack([np.hstack(tiles[:7]), np.hstack(tiles[7:])])
    cv2.imwrite(str(path), grid[:, :, ::-1])


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--out", default="results")
    ap.add_argument("--ckpt", default="checkpoints/disfa_adaptation_f0.pth")
    ap.add_argument("--detector", choices=["sfd", "haar", "none"], default="sfd",
                    help="sfd = same as original code (needs face-alignment); none = inputs are already face crops")
    ap.add_argument("--margin", type=float, default=0.0, help="extra margin around the face box (0.1 = 10%%)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()

    inp, out = Path(args.input), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in (inp.iterdir() if inp.is_dir() else [inp]) if p.suffix.lower() in IMG_EXT)

    model = AUHeatmapDetector(args.ckpt, device=args.device)
    det = FaceDetector(args.detector, model.device)

    with open(out / "intensities.csv", "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["image"] + [f"AU{a}" for a in AU_IDS])
        for p in files:
            bgr = cv2.imread(str(p))
            if bgr is None:
                print(f"skip (cannot read): {p}")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            box = det.largest_box(rgb)
            if box is None:
                print(f"skip (no face): {p}")
                continue
            face = square_crop(rgb, box, args.margin)
            heatmaps, intensities = model([face])
            heatmaps, intensities = heatmaps[0], intensities[0]

            np.save(out / f"{p.stem}_heatmaps.npy", heatmaps.astype(np.float32))
            if not args.no_png:
                save_overlay(face, heatmaps, intensities, out / f"{p.stem}_heatmaps.png")
            writer.writerow([p.name] + [f"{v:.4f}" for v in intensities])
            print(p.name, " ".join(f"AU{a}={v:.2f}" for a, v in zip(AU_IDS, intensities)))


if __name__ == "__main__":
    main()
