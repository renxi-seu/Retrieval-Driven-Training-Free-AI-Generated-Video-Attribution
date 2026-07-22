import os
import torch.multiprocessing as mp

mp.set_sharing_strategy('file_system')

# ---- Model & hardware ----

CHOICE_MODEL = "r3d18"    
GPU_ID = "0"

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

# ---- Scheme toggles ----

USE_SCHEME_1 = True  # multi-scale asymmetric quantization
USE_SCHEME_2 = True  # temporal residual gradient + RGB fusion

# ---- Dataset hyperparams ----

N_GALLERY, N_QUERY, N_BINARY = 100, 1000, 1000
NUM_FRAMES, IMG_SIZE = 64, 224
EVAL_KS = [10]

# ---- Quantization pairs ----

QUANT_PAIRS = [(0.5, 0.4), (1.2, 1), (1.9, 1.6), (2.6, 2.2)]
DEFAULT_QUANT = (1.0, 1.0)

# ---- Multi-modal weights ----

FEAT_WEIGHTS = (0.3, 0.7, 0)  # (residual, gradient, raw RGB)

# ---- Data paths ----

DATA_ROOT = "/path/to/GenVidBench_frames"
SUBSETS = ['hdvg', 'vript', 't2vz', 'ms', 'vc', 'pika', 'svd', 'musev', 'mora', 'cogvideo']

# ---- Feature persistence ----

FEATURE_DIR = "/path/to/features"
GALLERY_FEAT_FILE = "gallery.pt"


def get_class_label(subset_name: str) -> int:
    if subset_name in ['hdvg', 'vript']:
        return 0
    mapping = {
        't2vz': 1, 'ms': 2, 'vc': 3, 'pika': 4,
        'svd': 5, 'musev': 6, 'mora': 7, 'cogvideo': 8
    }
    return mapping[subset_name]


def is_real_subset(subset_name: str) -> bool:
    return subset_name in ['hdvg', 'vript']
