import argparse
import os
import random

import numpy as np
import torch

from config import (
    DATA_ROOT, SUBSETS, CHOICE_MODEL,
    N_GALLERY, N_QUERY, N_BINARY,
    USE_SCHEME_1, USE_SCHEME_2,
    FEATURE_DIR, GALLERY_FEAT_FILE,
    get_class_label, is_real_subset,
)
from transform import Transform
from dataset import get_model
from extract import extract_feat, save_features, load_features, eval_res


def _split_samples(subset_folders: dict, n_gallery: int, n_query: int, n_binary: int):
    """Split into gallery / query / binary sets (random shuffle each run)."""
    gal_s, que_s, bin_s = [], [], []
    for s in SUBSETS:
        folders = subset_folders.get(s, [])
        random.shuffle(folders)
        l = get_class_label(s)
        n_g = n_gallery if not is_real_subset(s) else max(n_gallery // 2, 1)
        gal_s.extend([(f, l, s) for f in folders[:n_g]])
        que_s.extend([(f, l, s) for f in folders[n_g: n_g + n_query]])
        bin_s.extend([(f, l, s) for f in folders[n_g + n_query: n_g + n_query + n_binary]])
    return gal_s, que_s, bin_s


def _build_label_map():
    label_to_name = {0: "Real (hdvg/vript)"}
    for s in SUBSETS:
        if not is_real_subset(s):
            label_to_name[get_class_label(s)] = s
    return label_to_name


def _report_attribution(q_f, q_l, g_f, g_l, label_to_name):
    """Source attribution: Rank-1 & mAP per class + overall."""
    print("\n" + "=" * 50)
    print("[Source Attribution  |  Rank-1 & mAP]")
    print("=" * 50)

    dist_q = (1 - torch.mm(q_f, g_f.t())).numpy()
    unique_labels = sorted(torch.unique(q_l).tolist())

    print(f"{'Class':<18} | {'Rank-1':<8} | {'mAP@10':<8}")
    print("-" * 38)

    for l in unique_labels:
        mask = (q_l == l).numpy()
        if not np.any(mask):
            continue
        name = label_to_name.get(l, f"Class {l}")
        r1, maps = eval_res(dist_q[mask], q_l[mask].numpy(), g_l.numpy())
        print(f"{name:<18} | {r1:.4f}   | {maps[10]:.4f}")

    ov_r1, ov_maps = eval_res(dist_q, q_l.numpy(), g_l.numpy())
    print("-" * 38)
    print(f"{'OVERALL':<18} | {ov_r1:.4f}   | {ov_maps[10]:.4f}")


def _report_detection(b_f, b_l, g_f, g_l, b_subs):
    """Binary detection: accuracy per subset + overall."""
    print("\n" + "=" * 50)
    print("[Binary Detection Accuracy]")
    print("=" * 50)

    dist_b = (1 - torch.mm(b_f, g_f.t())).numpy()
    nn_indices = np.argmin(dist_b, axis=1)
    preds_labels = g_l.numpy()[nn_indices]
    preds_binary = (preds_labels > 0).astype(int)

    gt_binary = np.array([1 if not is_real_subset(s) else 0 for s in b_subs])
    b_subs_np = np.array(b_subs)

    print(f"{'Subset':<18} | {'Count':<6} | {'Accuracy':<10}")
    print("-" * 38)

    for s in SUBSETS:
        mask = (b_subs_np == s)
        if not np.any(mask):
            continue
        acc = np.mean(preds_binary[mask] == gt_binary[mask])
        print(f"{s:<18} | {np.sum(mask):<6} | {acc:.4f}")

    ov_acc = np.mean(preds_binary == gt_binary)
    print("-" * 38)
    print(f"{'OVERALL':<18} | {len(b_subs):<6} | {ov_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="LPEA-based video source attribution & detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract only: customize gallery size, data root, output paths
  python test.py --extract-only --n-gallery 100 \\
      --data-root /data/videos --feature-dir ./features --gallery-file gallery.pt

  # Eval only: customize data root, query/binary counts, gallery path
  python test.py --eval-only --n-query 1000 --n-binary 1000 \\
      --data-root /data/videos --feature-dir ./features --gallery-file gallery.pt

  # Full pipeline: customize everything
  python test.py --n-gallery 100 --n-query 1000 --n-binary 1000 \\
      --data-root /data/videos
        """,
    )

    # ---- Mode flags ----
    parser.add_argument('--extract-only', action='store_true',
                        help='Extract & save gallery features, then exit')
    parser.add_argument('--eval-only', action='store_true',
                        help='Load pre-extracted gallery, extract query/binary on the fly, evaluate')

    # ---- Sample counts (override config) ----
    parser.add_argument('--n-gallery', type=int, default=None,
                        help=f'Gallery samples per subset (default: {N_GALLERY} from config)')
    parser.add_argument('--n-query', type=int, default=None,
                        help=f'Query samples per subset (default: {N_QUERY} from config)')
    parser.add_argument('--n-binary', type=int, default=None,
                        help=f'Binary samples per subset (default: {N_BINARY} from config)')

    # ---- Paths (override config) ----
    parser.add_argument('--data-root', type=str, default=None,
                        help=f'Root directory of frame datasets (default: {DATA_ROOT})')
    parser.add_argument('--feature-dir', type=str, default=None,
                        help=f'Directory to save/load features (default: {FEATURE_DIR})')
    parser.add_argument('--gallery-file', type=str, default=None,
                        help=f'Gallery feature filename (default: {GALLERY_FEAT_FILE})')

    args = parser.parse_args()

    # Resolve values: CLI arg > config default
    n_gallery = args.n_gallery if args.n_gallery is not None else N_GALLERY
    n_query   = args.n_query   if args.n_query   is not None else N_QUERY
    n_binary  = args.n_binary  if args.n_binary  is not None else N_BINARY
    data_root = args.data_root if args.data_root is not None else DATA_ROOT
    feat_dir  = args.feature_dir if args.feature_dir is not None else FEATURE_DIR
    gal_file  = args.gallery_file if args.gallery_file is not None else GALLERY_FEAT_FILE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = Transform(USE_SCHEME_1)
    model = get_model(CHOICE_MODEL).to(device).eval()
    gallery_path = os.path.join(feat_dir, gal_file)

    print("--- LPEA-based Adaptive Video Source Attribution & Detection ---")
    print(f"Data root:     {data_root}")
    print(f"Feature dir:   {feat_dir}")
    print(f"Gallery file:  {gal_file}")
    print(f"N_gallery: {n_gallery}, N_query: {n_query}, N_binary: {n_binary}")

    subset_folders = {
        s: [f.path for f in os.scandir(os.path.join(data_root, s)) if f.is_dir()]
        for s in SUBSETS if os.path.exists(os.path.join(data_root, s))
    }

    if not any(subset_folders.values()):
        found = [s for s in SUBSETS if os.path.exists(os.path.join(data_root, s))]
        print(f"ERROR: No video folders found under data_root: {data_root}")
        if found:
            print(f"  Existing subdirs: {found}")
            for s in found:
                n = len([f for f in os.scandir(os.path.join(data_root, s)) if f.is_dir()])
                print(f"    {s}: {n} folders")
        else:
            print(f"  None of the expected subsets {SUBSETS} exist under this path.")
            print(f"  Use --data-root to specify the correct dataset path.")
        return

    if args.extract_only:
        gal_s, _, _ = _split_samples(subset_folders, n_gallery, n_query, n_binary)
        print(f"Gallery samples: {len(gal_s)}")
        g_f, g_l, _ = extract_feat(gal_s, transform, model, device, desc="Extracting Gallery")
        save_features(g_f, g_l, [], gallery_path)
        print(f"Gallery features saved to: {gallery_path}")
        print(f"  Feature dims: {g_f.shape}")
        return

    if args.eval_only:
        print(f"Loading gallery features from: {gallery_path}")
        g_f, g_l, _ = load_features(gallery_path)
        print(f"  Gallery dims: {g_f.shape}")

        gal_s, que_s, bin_s = _split_samples(subset_folders, n_gallery, n_query, n_binary)
        print(f"Query: {len(que_s)}, Binary: {len(bin_s)}")

        q_f, q_l, q_subs = extract_feat(que_s, transform, model, device, desc="Extracting Query")
        b_f, b_l, b_subs = extract_feat(bin_s, transform, model, device, desc="Extracting Binary")
    else:
        gal_s, que_s, bin_s = _split_samples(subset_folders, n_gallery, n_query, n_binary)
        print(f"Gallery: {len(gal_s)}, Query: {len(que_s)}, Binary: {len(bin_s)}")

        g_f, g_l, _ = extract_feat(gal_s, transform, model, device, desc="Extracting Gallery")
        q_f, q_l, q_subs = extract_feat(que_s, transform, model, device, desc="Extracting Query")
        b_f, b_l, b_subs = extract_feat(bin_s, transform, model, device, desc="Extracting Binary")

    label_to_name = _build_label_map()
    _report_attribution(q_f, q_l, g_f, g_l, label_to_name)
    _report_detection(b_f, b_l, g_f, g_l, b_subs)


if __name__ == "__main__":
    main()
