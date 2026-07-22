# Retrieval-Driven Training-Free AI-Generated Video Attribution

<img width="5120" height="1800" alt="39456" src="https://raw.githubusercontent.com/renxi-seu/Images/refs/heads/main/training-free%20video%20attribution.jpg" />
Retrieval-Driven Training-Free AI-Generated Video Attribution, arXiv

## Dataset
[GenVidBench](https://github.com/genvidbench/) is a comprehensive and recently introduced benchmark for AI-generated video detection. It contains 100,000 semantic labels, along with the original prompts and images used during the generation process. The dataset combines real-world videos from HD-VG and Vript with synthetic videos produced by eight generators: T2V-Zero (T2VZ), ModelScope (MS), VideoCrafter2 (VC2), Pika, SVD, MuseV, Mora, and CogVideo (CogV). In our setting, we combine HD-VG and Vript into a single Real category.
Before execution, you should first save each video in GenVidBench frame by frame as a .jpg image, and save all images of each video as a separate folder, organized according to the original structure of GenVidBench, and finally form the following GenVidBench_frames:
```
GenVidBench_frames/
├── hdvg/
│   ├── 031hrZOby7s/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   ├── -05SQDsPtUQ/
│   ├── 0BkSXurscOI/
│   └── ...
├── vript/
├── t2vz/
├── ms/ 
├── vc/
├── pika/
├── svd/
├── musev/
├── mora/
└── cogvideo/
```

## Construct the Database
You can construct the referencing database of GenVidBench (1, 10, 100-shot) and save it in the specified location by running the following command:
```
python test.py --extract-only \\
               --n-gallery 1/10/100 \\
               --data-root /path/to/GenVidBench_frames \\
               --feature-dir /path/to/features \\
               --gallery-file gallery.pt
```

## Evaluation
You can evaluate our approach on GenVidBench with Rank-1 and mAP reported by running the following command:
```
python test.py --eval-only \\
               --n-query 1000 \\
               --n-binary 1000 \\
               --data-root /path/to/GenVidBench_frames \\
               --feature-dir /path/to/features \\
               --gallery-file gallery.pt
```

```

```

