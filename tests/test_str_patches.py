import math

import pytest
from PIL import Image

from cs_aualign_vit.data.str_patches import compose, compute_rois, validate_rois

LANDMARKS = [[30, 30], [70, 30], [50, 50], [35, 75], [65, 75]]


def test_rois_use_eyes_and_mouth_corners_with_rounded_side():
    rois = compute_rois(LANDMARKS, (100, 100))
    assert rois[0] == (22, 22, 38, 38)  # d_io=40 -> s=16, centre (30, 30)
    assert rois[2] == (27, 67, 43, 83)
    assert all(r[2] - r[0] == 16 == r[3] - r[1] for r in rois)


def test_odd_side_uses_floor_half():
    rois = compute_rois([[30, 30], [62, 30], [0, 0], [35, 75], [65, 75]], (100, 100))
    s = round(0.4 * 32)
    assert s == 13 and rois[0][0] == 30 - 6 and rois[0][2] - rois[0][0] == 13


@pytest.mark.parametrize(
    "landmarks",
    [
        [[math.nan, 0]] * 5,
        [[0, 0], [math.inf, 0], [0, 0], [0, 0], [0, 0]],
        [[10, 10], [11, 10], [0, 0], [10, 20], [20, 20]],  # s < 1
        [[1, 2]],
        None,
    ],
)
def test_invalid_landmarks(landmarks):
    assert validate_rois(compute_rois(landmarks, (100, 100)), (100, 100)) == "invalid_landmark"


def test_out_of_bounds_is_rejected_not_clipped():
    rois = compute_rois([[2, 30], [42, 30], [0, 0], [35, 75], [65, 75]], (100, 100))
    assert validate_rois(rois, (100, 100)) == "out_of_bounds"
    assert validate_rois([(0, 0, 16, 16), (16, 0, 32, 16)], (32, 16)) is None  # touching edges


def test_overlap_rejected_but_touching_is_allowed():
    assert validate_rois([(0, 0, 10, 10), (9, 9, 19, 19)], (100, 100)) == "overlap"
    assert validate_rois([(0, 0, 10, 10), (10, 0, 20, 10)], (100, 100)) is None


def test_compose_pastes_four_regions_on_a_copy():
    source = Image.new("RGB", (100, 100), (0, 0, 0))
    target = Image.new("RGB", (60, 60), (255, 0, 0))
    rs = compute_rois(LANDMARKS, (100, 100))
    rt = [(0, 0, 8, 8), (10, 0, 18, 8), (20, 0, 28, 8), (30, 0, 38, 8)]
    out = compose(source, target, rs, rt)
    assert out is not source and source.getpixel((30, 30)) == (0, 0, 0)
    for box in rs:
        assert out.getpixel((box[0] + 3, box[1] + 3)) == (255, 0, 0)
    assert out.getpixel((50, 50)) == (0, 0, 0)
