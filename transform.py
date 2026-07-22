"""
LPEA (Local Prediction Error Alignment) adaptive transform.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from config import IMG_SIZE, QUANT_PAIRS, DEFAULT_QUANT


class Transform:

    def __init__(self, use_scheme_1: bool):
        self.use_scheme_1 = use_scheme_1
        self.pairs = QUANT_PAIRS if use_scheme_1 else [DEFAULT_QUANT]
        self.rgb_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _get_adaptive_m(self, x_tensor: torch.Tensor):

        device = x_tensor.device
        img = x_tensor.permute(2, 0, 1).unsqueeze(0)

        kernel = torch.ones((3, 1, 3, 3), device=device) / 9.0
        local_mean = F.conv2d(img, kernel, padding=1, groups=3)

        prediction_error = (img - local_mean).squeeze(0).view(3, -1)
        cov = torch.mm(prediction_error, prediction_error.t()) / (prediction_error.size(1) - 1)
        _, V = torch.linalg.eigh(cov)
        v_main = torch.abs(V[:, -1])

        w_raw = v_main / (v_main.sum() + 1e-8)
        if w_raw[1] <= 0.505:
            wg = torch.tensor(0.51, device=device)
            rem = 1.0 - wg
            s_rb = w_raw[0] + w_raw[2] + 1e-8
            wr = rem * (w_raw[0] / s_rb)
            wb = rem * (w_raw[2] / s_rb)
            y_row = torch.stack([wr, wg, wb])
        else:
            y_row = w_raw

        def norm_vec(v):
            return v / (torch.norm(v) + 1e-8)

        def proj(target, base):
            return (torch.dot(target, base) / torch.dot(base, base)) * base

        y = y_row
        u_init = torch.tensor([1.0, 0.0, -1.0], device=device)
        u = norm_vec(u_init - proj(u_init, y))
        v_init = torch.tensor([1.0, -1.0, 0.0], device=device)
        v = norm_vec(v_init - proj(v_init, y) - proj(v_init, u))

        m_t = torch.stack([y, u, v])
        return m_t, torch.inverse(m_t)

    def __call__(self, img):
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        rgb_tensor = self.rgb_norm(img_resized)
        x = torch.from_numpy(np.array(img_resized)).float()

        m_t, m_inv = self._get_adaptive_m(x)
        mapped_space = torch.matmul(x, m_t.t())

        results = []
        for s_y, s_uv in self.pairs:
            q_space = mapped_space.clone()
            q_space[..., 0] = torch.round(mapped_space[..., 0] / s_y) * s_y
            q_space[..., 1:] = torch.round(mapped_space[..., 1:] / s_uv) * s_uv
            x_recon = torch.matmul(q_space, m_inv.t())
            res = x - x_recon
            strength = torch.std(res).item()
            res = (res - res.min()) / (res.max() - res.min() + 1e-6)
            results.append((res.permute(2, 0, 1), strength))

        return results, rgb_tensor
