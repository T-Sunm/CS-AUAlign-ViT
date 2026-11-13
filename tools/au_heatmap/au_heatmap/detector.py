"""Load disfa_adaptation_f0.pth and return 12 AU heatmaps + intensities."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import FANAU

# DISFA AU order used by the checkpoint (channel i -> AU_NAMES[i])
AU_IDS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
AU_NAMES = [
    "Inner_Brow_Raiser", "Outer_Brow_Raiser", "Brow_Lowerer", "Upper_Lid_Raiser",
    "Cheek_Raiser", "Nose_Wrinkler", "Lip_Corner_Puller", "Lip_Corner_Depressor",
    "Chin_Raiser", "Lip_Stretcher", "Lips_Part", "Jaw_Drop",
]


def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}
    model = FANAU(n_points=12)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. missing={missing[:5]}... unexpected={unexpected[:5]}...")
    return model.eval().to(device)


def preprocess(face_rgb_uint8):
    """HxWx3 uint8 RGB face crop -> 1x3x256x256 float in [0,1] (min-max, as in the original code)."""
    t = torch.from_numpy(np.array(face_rgb_uint8, copy=True, order="C")).permute(2, 0, 1).float()[None]
    t = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)
    t = (t - t.min()) / (t.max() - t.min() + 1e-8)
    return t


class AUHeatmapDetector:
    def __init__(self, ckpt_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(ckpt_path, self.device)

    @torch.no_grad()
    def __call__(self, faces):
        """
        faces: tensor Bx3x256x256 in [0,1], or a list of HxWx3 uint8 RGB face crops.
        returns heatmaps (B x 12 x 64 x 64, numpy) and intensities (B x 12, numpy; peak of each map).
        """
        if not torch.is_tensor(faces):
            faces = torch.cat([preprocess(f) for f in faces], 0)
        heatmaps = self.model(faces.to(self.device))
        intensities = heatmaps.amax(dim=(2, 3))
        return heatmaps.cpu().numpy(), intensities.cpu().numpy()
