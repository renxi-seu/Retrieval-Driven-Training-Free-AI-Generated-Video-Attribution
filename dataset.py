import glob
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

from config import NUM_FRAMES, IMG_SIZE, QUANT_PAIRS, USE_SCHEME_1


class GenVidDataset(Dataset):
    """
    Per-sample pipeline:
    1. Sample NUM_FRAMES frames (palindrome-pad if short)
    2. Apply PiDTransform per frame → multi-scale residuals
    3. (optional) Compute temporal gradient (Scheme 2)
    """

    def __init__(self, samples, transform, use_scheme_2: bool):
        self.samples = samples
        self.transform = transform
        self.use_scheme_2 = use_scheme_2

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, subset = self.samples[idx]
        frames = sorted(glob.glob(os.path.join(path, "*.jpg")))
        num_res = len(QUANT_PAIRS) if USE_SCHEME_1 else 1

        if not frames:
            return [
                (torch.zeros(3, NUM_FRAMES, IMG_SIZE, IMG_SIZE),
                 torch.zeros(3, NUM_FRAMES, IMG_SIZE, IMG_SIZE),
                 torch.zeros(3, NUM_FRAMES, IMG_SIZE, IMG_SIZE), 0.0)
            ] * num_res, label, subset

        # palindrome padding
        res_frames = frames.copy()
        db = True
        while len(res_frames) < NUM_FRAMES:
            res_frames.extend(frames[::-1] if db else frames)
            db = not db
        selected = res_frames[:NUM_FRAMES]

        buckets = [[] for _ in range(num_res)]
        strengths = [0.0 for _ in range(num_res)]
        rgb_bucket = []

        for p in selected:
            img = Image.open(p).convert('RGB')
            res_list, rgb_t = self.transform(img)
            rgb_bucket.append(rgb_t)
            for i in range(num_res):
                buckets[i].append(res_list[i][0])
                strengths[i] += res_list[i][1]

        rgb_video = torch.stack(rgb_bucket, dim=1)

        results = []
        for i in range(num_res):
            r_t = torch.stack(buckets[i], dim=1)
            delta_r = torch.zeros_like(r_t)
            if self.use_scheme_2:
                diff = r_t[:, 1:, :, :] - r_t[:, :-1, :, :]
                delta_r = torch.cat([diff, diff[:, -1:, :, :]], dim=1)
            results.append((r_t, delta_r, rgb_video, strengths[i] / NUM_FRAMES))

        return results, label, subset


def get_model(name: str) -> nn.Module:
    if name == "r3d18":
        from torchvision.models.video import r3d_18, R3D_18_Weights
        m = r3d_18(weights=R3D_18_Weights.DEFAULT)
        m.fc = nn.Identity()
    elif name == "swin":
        from torchvision.models.video import swin3d_s, Swin3D_S_Weights
        m = swin3d_s(weights=Swin3D_S_Weights.DEFAULT)
        m.head = nn.Identity()
    elif name == "i3d":
        from torchvision.models.video import s3d, S3D_Weights
        m = s3d(weights=S3D_Weights.DEFAULT)
        m.classifier = nn.Identity()
    else:
        raise ValueError(f"Unknown model: {name}")
    return m
