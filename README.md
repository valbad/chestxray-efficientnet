# ChestXRay EfficientNet-B4 Classifier

This repository provides a **multi-class chest X-ray classifier** built on **EfficientNet-B4**, trained and evaluated on the dataset from:
> * **Kun-Hsing Yu. “BMI707 Assignment 2 Q5.” Kaggle (2025).** https://kaggle.com/competitions/2025bmi707-assignment-2-q5
The task consists of predicting **three classes** from frontal chest radiographs.
The project includes:
- A complete data-handling pipeline
- Training with **Focal Loss, class weighting, and mixed-precision (AMP)**
- Clean dataset download utilities
- **Test-Time Augmentation (TTA)** inference
- Scripts for **CLI data loading**, **CLI training** and **CLI inference**
- A modular Python API (```src/```)
- A reproducible Jupyter notebook (```notebooks/```)

---
## 1. Project Structure
```kotlin
chestxray-efficientnet/
│
├── assets/
│   ├── example_predictions/
│   └── figures/
│       ├── val/confusion_matrix.png
│       └── val_tta/confusion_matrix.png
│
├── configs/
│   └── default.yaml
│
├── data/
│   ├── train/
│   ├── testPublic/
│   └── train.csv
│
├── notebooks/
│   └── chestxray_training_and_inference.ipynb
│
├── outputs/
│   └── models/
│       ├── efficientnet_b2_best.safetensors
│       └── efficientnet_b2_last_epoch.safetensors
│
├── scripts/
│   ├── download_dataset.py
│   ├── run_training.py
│   └── run_inference.py
│
├── src/
│   ├── data/
│   ├── inference/
│   ├── models/
│   └── training/
│
├── requirements.txt
└── LICENSE (MIT)
```

---
## 2. Configuration (default.yaml)
```yaml
model:
  name: "efficientnet_b4"
  pretrained: True
  num_classes: 3

data:
  data_dir: "data"
  train_csv: "data/train.csv"

  img_size: 380
  batch_size: 32
  num_workers: 8
  val_size: 0.15
  random_state: 42

  dropbox_train_tar: "https://www.dropbox.com/s/feaakvusdvkep3n/train.tar"
  dropbox_train_csv: "https://www.dropbox.com/s/9kebfecemhfkj7k/train.csv"
  dropbox_test_tar:  "https://www.dropbox.com/s/20d8b9z8znc2lmp/testPublic.tar"
  suffix: "?dl=1"

training:
  epochs: 20
  lr: 1e-4
  weight_decay: 1e-5
  early_stopping_patience: 5
  device: "cuda"

paths:
  output_dir: "outputs"

```

---
## 3. Dataset

The dataset contains chest radiographs labeled into **three classes.**

It was originally part of a Kaggle educational assignment and derived from a broader study involving automatic X-ray analysis from hospital record. From this regard, note that the quality of the classification labels is quite limited. 

The dataset is downloaded automatically if the ```data``` does not contain a train or test directory (the procedure looks for keywords ```train``` and ```test``` in the folder).
Raw files come from the Dropbox URLs defined in the config above.

--- 
## 4. Training
### Run from CLI
```bash
python -m scripts.run_training \
    --config configs/default.yaml \
    --train_dir data/train \
    --models_dir outputs/model/ \
    --history_file outputs/history/training_history.csv
```
This will:

- Train EfficientNet-B4 with Focal Loss

- Save best and last models'weights at ```outputs/model/efficientnet_b4_best.safetensors``` and ```outputs/model/efficientnet_b4_last_epoch.safetensors```

- Save history (Training and Validation Loss and Accuracies) at ```outputs/history/training_history.csv```

AMP (mixed precision) is enabled for speed.

---
## 5. Inference

Two inference modes are provided:
### (A) Standard Inference

```bash
python -m scripts.run_inference \
    --model_path outputs//models/efficientnet_b4_best.safetensors \
    --config configs/default.yaml \
    --mode normal \
    --test_dir data/testPublic \
    --save_dir outputs/predictions/normal/
```
Output: ```predictions.csv```

### (B) TTA (Test Time Augmentation)

```bash
python -m scripts.run_inference \
    --model_path outputs//models/efficientnet_b4_best.safetensors \
    --config configs/default.yaml \
    --mode tta \
    --test_dir data/testPublic \
    --save_dir outputs/predictions/normal/
```
Output: ```predictions_tta.csv```

---
## 6. Validation Performance

### Without TTA

- Accuracy: 0.73

- AUC (micro): 0.879

- Confusion matrix saved at: assets/figures/val/confusion_matrix.png

### With TTA

- Accuracy: 0.73

- AUC (micro): 0.881

- Confusion matrix saved at: assets/figures/val_tta/confusion_matrix.png

TTA gives a **small but consistent improvement.**

---
## 7. Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

---
## 8. License

This project is released under the MIT License.
See LICENSE for details.

---

## 9. Citation

If you use this code or reproduce the benchmark, please cite:

> Kun-Hsing Yu. BMI707 Assignment 2 Q5. Kaggle, 2025. https://kaggle.com/competitions/2025bmi707-assignment-2-q5
