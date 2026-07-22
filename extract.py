import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import EVAL_KS, FEAT_WEIGHTS, USE_SCHEME_1, USE_SCHEME_2
from dataset import GenVidDataset


# ---- Persistence ----

def save_features(feats: torch.Tensor, labels: torch.Tensor, subs: list,
                  filepath: str):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    torch.save({'feats': feats, 'labels': labels, 'subs': subs}, filepath)


def load_features(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Feature file not found: {filepath}\n"
            f"Run first: python test.py --extract-only"
        )
    data = torch.load(filepath, map_location='cpu', weights_only=False)
    return data['feats'], data['labels'], data['subs']


# ---- Metrics ----

def eval_res(dist_mat: np.ndarray, q_labels: np.ndarray, g_labels: np.ndarray):
    """
    Returns:
        rank1:   Rank-1 accuracy
        ap_dict: {k: mAP@k}
    """
    r1 = 0
    aps = {k: [] for k in EVAL_KS}

    for i in range(len(q_labels)):
        idx = np.argsort(dist_mat[i])
        if g_labels[idx[0]] == q_labels[i]:
            r1 += 1
        for k in EVAL_KS:
            top_k = idx[:k]
            match = (g_labels[top_k] == q_labels[i]).astype(float)
            if match.sum() > 0:
                prec = np.cumsum(match) / np.arange(1, k + 1)
                aps[k].append(np.sum(prec * match) / match.sum())
            else:
                aps[k].append(0.0)

    return r1 / len(q_labels), {k: np.mean(aps[k]) for k in EVAL_KS}


# ---- Extraction pipeline ----

def extract_feat(samples, transform, model, device, desc: str = ""):

    loader = DataLoader(
        GenVidDataset(samples, transform, USE_SCHEME_2),
        batch_size=1,
        num_workers=4
    )

    feats, labels, subs = [], [], []
    wr, wg, wb = FEAT_WEIGHTS

    for d_list, y, s_n in tqdm(loader, desc=desc):
        with torch.no_grad():
            sc_feats, sc_strs = [], []

            for r_t, d_r, rgb_v, sv in d_list:
                f_res = F.normalize(model(r_t.to(device)), p=2, dim=1)

                if USE_SCHEME_2:
                    f_grad = F.normalize(model(d_r.to(device)), p=2, dim=1)
                    f_rgb = F.normalize(model(rgb_v.to(device)), p=2, dim=1)
                    f_combined = wr * f_res + wg * f_grad + wb * f_rgb
                    sc_feats.append(F.normalize(f_combined, p=2, dim=1))
                else:
                    sc_feats.append(f_res)

                sc_strs.append(sv.item())

            if USE_SCHEME_1:
                w_sc = F.softmax(torch.tensor(sc_strs).to(device), dim=0)
                final_f = sum(w_sc[i] * sc_feats[i] for i in range(len(sc_feats)))
            else:
                final_f = sc_feats[0]

            feats.append(F.normalize(final_f, p=2, dim=1).cpu())
            labels.append(y)
            subs.append(s_n[0])

    return torch.cat(feats), torch.cat(labels), subs
