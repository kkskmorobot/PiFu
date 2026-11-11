# PiFuV2 data preparation

`prepare_skin7_dataset.py` builds a new YOLO segmentation dataset without changing the original files under `Data/train`.

## Strategy

- Split by `lesion_id`, so photographs of the same lesion cannot cross train/val/test.
- Approximately preserve the seven-class distribution in every split.
- Keep validation and test images untouched.
- Cap training `nv` at approximately 3,500 images while keeping lesion groups intact.
- Augment minority classes only, with a maximum fourfold expansion per class.
- Apply matching flips/rotations to images and masks plus mild image-only colour jitter.
- Convert binary masks into YOLO segmentation polygons.
- Refuse to overwrite an existing output directory.

## Environment

After PyCharm repairs or recreates the virtual environment, install:

```powershell
pip install numpy opencv-python ultralytics
```

## Preview the split and balance plan

Run this first. It writes nothing:

```powershell
python prepare_skin7_dataset.py
```

## Generate the dataset

```powershell
python prepare_skin7_dataset.py --apply
```

The result is written to `Data/processed_skin7`, including:

- `images/{train,val,test}`
- `labels/{train,val,test}`
- `metadata/{train,val,test}.csv`
- `data.yaml`
- `summary.json`

The official ISIC Task 1 and Task 3 test downloads are intentionally not used here. The internal test split is made from HAM10000 because those samples have both disease labels and lesion masks.
