# Retrieval-Driven-Training-Free-AI-Generated-Video-Attribution

<img width="5120" height="1800" alt="39456" src="https://raw.githubusercontent.com/renxi-seu/Images/refs/heads/main/training-free%20video%20attribution.jpg" />
Retrieval-Driven Training-Free AI-Generated Video Attribution, arXiv

## Dataset
[GenVidBench](https://github.com/genvidbench/) is a comprehensive and recently introduced benchmark for AI-generated video detection. It contains 100,000 semantic labels, along with the original prompts and images used during the generation process. The dataset combines real-world videos from HD-VG and Vript with synthetic videos produced by eight generators: T2V-Zero (T2VZ), ModelScope (MS), VideoCrafter2 (VC2), Pika, SVD, MuseV, Mora, and CogVideo (CogV). In our setting, we combine HD-VG and Vript into a single Real category.

## Construct the Database
You can construct the referencing database of GenVidBench (1, 10, 100-shot) and save it in the specified location by running the following command:
```
python feature.py --dataset_path='/path/to/dataset'
                  --weight_path='path/to/weights.pth'
                  --save_dir='path/to/features'
                  --num_gallery_per_class=1/5/10
                  --mode='full/patch'
                  --patch_mode='random/max/min'
                  --patch_size=32
```

## Evaluation
You can evaluate our approach on GenVidBench with Rank-1 and mAP reported by running the following command:
```
python attribute.py --dataset_path='/path/to/dataset'
                    --weight_path='path/to/weights.pth'
                    --save_dir='path/to/features'
                    --num_query_per_class=num
                    --mode='full/patch'
                    --patch_mode='random/max/min'
                    --patch_size=32
```

```

```

