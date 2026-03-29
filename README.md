# neuralink-bc

A multilayer perceptron built from scratch in Python for binary classification of breast cancer tumors (malignant vs. benign) on the [Wisconsin Diagnostic Breast Cancer dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).

No ML framework — only NumPy. The network implements forward propagation, backpropagation, mini-batch SGD, and three-valued output reporting.

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | ~97% |
| Cross-val Mean (10-fold) | ~96–97% |
| Loss function | MSE (training) / BCE (evaluation) |
| Output layer | Softmax |

---

## Architecture

```
src/
├── network.py         # Network class: forward pass, backprop, SGD, save/load
├── data.py            # Data loading, preprocessing, normalization, EDA tools
├── main.py            # Training entry point (CLI)
├── predict.py         # Evaluation entry point (CLI)
├── split.py           # Train/test CSV split utility (CLI)
└── cross_validate.py  # Stratified k-fold cross-validation
```

### Network

- **Input**: 30 features (standardized)
- **Hidden layers**: configurable via `--layer` (default: 24 × 24 × 24)
- **Output**: 2 neurons (Benign / Malignant) with softmax
- **Activation**: sigmoid (hidden), softmax (output)
- **Optimizer**: mini-batch SGD
- **Loss**: MSE for training, BCE for final evaluation

### Data flow

```
data.csv
   │
   ▼
split.py           → training_data.csv / test_data.csv
   │
   ▼
main.py            → trains network, saves model to model/best_model.pkl
   │
   ▼
predict.py         → loads model, evaluates on test set
```

---

## Getting Started

### Requirements

```
numpy
pandas
scikit-learn
matplotlib
seaborn
```

Install:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### 1. Split the dataset

```bash
cd src
python split.py --csv ../data/data.csv --ratio 0.3 --out ../data
```

### 2. Train

```bash
python main.py \
  --train_set ../data/training_data.csv \
  --test_set ../data/test_data.csv \
  --layer 24 24 \
  --epochs 100 \
  --learning_rate 0.1 \
  --batch_size 16
```

### 3. Evaluate

```bash
python predict.py \
  --model ../model/best_model.pkl \
  --train_set ../data/training_data.csv \
  --test_set ../data/test_data.csv
```

### 4. Cross-validate

```bash
python cross_validate.py
```

---

## Training Curves

![Training curves](assets/training_curves.png)

Loss converges smoothly with no overfitting. Test accuracy stabilizes around 97% after ~30 epochs.

---

## Dataset

[UCI Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) — 569 samples, 30 real-valued features computed from digitized FNA images of breast masses.

- **M** (malignant): 212 samples
- **B** (benign): 357 samples

---

## Project Structure

```
neuralink-bc/
├── data/
│   └── data.csv
├── model/              # saved .pkl models (git-ignored)
├── assets/
│   └── training_curves.png
├── src/
│   ├── network.py
│   ├── data.py
│   ├── main.py
│   ├── predict.py
│   ├── split.py
│   └── cross_validate.py
├── .gitignore
└── README.md
```