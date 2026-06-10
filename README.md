# A Warping Synthetic Minority Oversampling Technique for Imbalanced Time Series Classification

A Python implementation of **Safe-Level SMOTE** adapted for time series data, using **Dynamic Time Warping (DTW)** as the distance metric and a **path-aware interpolation** strategy to generate high-quality synthetic minority-class samples.

---

## Project Structure

```
.
├── main.py          
├── WSMOTE.py        # Core algorithm
├── data_loader.py   # Dataset loading and utility helpers
├── requirements.txt # Python dependencies
└── README.md
```
---

## Installation

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Edit the data path

Open `main.py` and update `train_path` in the `main()` function to point to your dataset:

```python
train_path = r"path/to/your/dataset_TRAIN.txt"
```

### Run

```bash
python main.py
```

The script will:

1. Load and analyse the dataset class distribution.
2. Generate synthetic minority-class samples.
3. Save the balanced dataset to `ECG200_WSMOTE.npz`.
4. Print a before/after class distribution summary.

### Load the balanced dataset later

```python
import numpy as np

data = np.load("ECG200_WSMOTE.npz")
X_train = data["X_train"]
y_train = data["y_train"]
```