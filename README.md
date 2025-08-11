# 3D-GCN Classification

This repository contains an implementation of a **3D Point Cloud Classification using Graph
Convolutional Network with Multi-Scale Pooling** for classifying 3D point cloud objects from ModelNet10 dataset.

---

## 📂 Project Structure
```bash 
3dgcn/
├── configs/
│ ├── config.yaml
│ ├── model.yaml
│ ├── train.yaml 
│ └── data.yaml 
│
├── src/
│ ├── models/
│ │ └── gcn.py
│ ├── metrics/
│ │ └── evaluation.py
│ ├── utils/
│ │ ├── data_loader.py
| | ├── data_utils.py
│ │ └── graph_utils.py
│ └── training.py # main training script
│
├── requirements.txt
├── .gitignore
└── README.md

```

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/3dgcn.git
cd 3dgcn
```

### 2. Create environment
```bash
conda create -n gcn_hydra python=3.10
conda activate gcn_hydra
```

### 3. Install requirements
```bash
pip install -r requirements.txt
```

## 📊 Dataset

This project uses the **[ModelNet10 dataset](https://modelnet.cs.princeton.edu/)**.

### 1. Create a `data/` directory
In the project root:
```bash
mkdir -p data/modelnet10
```
### 2. Download ModelNet10
Download the dataset from:
👉 https://modelnet.cs.princeton.edu/
Unzip it and place it inside `data/modelnet10`, so you get a structure like:
```bash
data/modelnet10/
├── bathtub/
│   ├── train/
│   └── test/
├── bed/
│   ├── train/
│   └── test/
├── chair/
│   ├── train/
│   └── test/
...
```
### 3. Configure paths
Make sure `configs/data.yaml` points to the correct directories.

## 🚀 Training
Run training with Hydra:
```bash
python -m src.training
```

You can override config values, for example: 
```bash
python -m src.training train.epochs=50 train.batch_size=16 model.hidden_dim=128
```

## 📈 Evaluation
After training, evaluation runs automatically:
*Confusion matrices (absolute + percentage)* and
*Classification report (precision/recall/F1)*

They are saved in `evaluation/`:
```bash
evaluation/
│── confusion_matrix_absolute_with_relabel.png
│── confusion_matrix_percentage_with_relabel.png
│── classification_report_with_relabel.txt
│── confusion_matrix_absolute_no_relabel.png
│── confusion_matrix_percentage_no_relabel.png
│── classification_report_no_relabel.txt
```
