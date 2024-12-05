# Automatic Number Plate Recognition
This project implements an Automatic Number Plate Recognition (ANPR) system using the [UC3M-LP](https://github.com/ramajoballester/UC3M-LP) dataset, providing tools for number plate detection, recognition, and evaluation.

## Instruction
### Data
1. Download the dataset [here](https://doi.org/10.21950/OS5W4Z)
2. Run the script to transform the dataset to YOLO format. It creates two versions of the dataset, we only used the high-res one. 
```
python scripts/labels2yolo.py path/to/UC3M-LP 320 160
```
3. Run the script to reproduce the train-val-test split for the original images.
```
python scripts/split_train_val.py
```
### Recognition
Run ```baseline.py``` to get metrics for [EasyOCR](https://github.com/JaidedAI/EasyOCR) or [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr). Example:
```
python baseline.py --split val --engine easyocr
```