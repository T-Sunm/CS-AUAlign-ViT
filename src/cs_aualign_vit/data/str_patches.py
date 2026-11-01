"""Pure geometry and pixel helpers for Self-Training Resampling patches."""

import math

from PIL import Image

# Left eye, right eye, left/right mouth corner (MTCNN order; the nose, index 2, is unused).
LANDMARK_INDICES = (0, 1, 3, 4)
ROI_SCALE = 0.4


def compute_rois(landmarks, size):
    """Return four half-open ``(left, top, right, bottom)`` boxes, or ``None`` if invalid."""
    try:
        points = [(float(landmarks[i][0]), float(landmarks[i][1])) for i in LANDMARK_INDICES]
    except (TypeError, IndexError, ValueError):
        return None
    if not all(math.isfinite(v) for point in points for v in point):
        return None
    side = round(ROI_SCALE * math.dist(points[0], points[1]))
    if side < 1:
        return None
    rois = []
    for x, y in points:
        left, top = round(x) - side // 2, round(y) - side // 2
        rois.append((left, top, left + side, top + side))
    return rois


def validate_rois(rois, size):
    """Return ``None`` when usable, else ``invalid_landmark``/``out_of_bounds``/``overlap``."""
    if rois is None:
        return "invalid_landmark"
    width, height = size
    for left, top, right, bottom in rois:
        if left < 0 or top < 0 or right > width or bottom > height:
            return "out_of_bounds"
    for i, a in enumerate(rois):
        for b in rois[i + 1 :]:
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                return "overlap"
    return None


def compose(source, target, rois_source, rois_target):
    """Paste four target patches, bilinearly resized, onto a copy of the source image."""
    if len(rois_source) != len(rois_target):
        raise ValueError("Source and target must have the same number of ROIs")
    result = source.copy()
    for box_s, box_t in zip(rois_source, rois_target, strict=True):
        patch = target.crop(box_t).resize(
            (box_s[2] - box_s[0], box_s[3] - box_s[1]), Image.Resampling.BILINEAR
        )
        result.paste(patch, (box_s[0], box_s[1]))
    return result
