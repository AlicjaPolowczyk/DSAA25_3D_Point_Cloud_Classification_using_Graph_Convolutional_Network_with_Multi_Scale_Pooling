# 3D Point Cloud Classification using Graph Convolutional Network with Multi-Scale Pooling
**Authors:**
  - [Alicja Polowczyk](https://orcid.org/0009-0001-3110-8255)
  - [Agnieszka Polowczyk](https://orcid.org/0009-0008-1583-4493)
  - [Antoni Jaszcz](https://orcid.org/0000-0002-8997-0331)
  - [Marcin Woźniak](https://orcid.org/0000-0002-9073-5347)
  - [Dawid Połap](https://orcid.org/0000-0003-1972-5979)



## Overview
This repository contains the source code related to the paper:

**_3D Point Cloud Classification using Graph Convolutional Network with Multi-Scale Pooling_**

to be presented at the **DSAA 2025 (IEEE International Conference on Data Science and Advanced Analytics)**.

---

## 📂 Project Structure
```bash 
DSAA25_3D_Point_Cloud_Classification_using_Graph_Convolutional_Network_with_Multi_Scale_Pooling/
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
git clone https://github.com/AlicjaPolowczyk/DSAA25_3D_Point_Cloud_Classification_using_Graph_Convolutional_Network_with_Multi_Scale_Pooling

cd DSAA25_3D_Point_Cloud_Classification_using_Graph_Convolutional_Network_with_Multi_Scale_Pooling
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

## License
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)\
This work is licensed under the Creative Commons License.\
Feel free to use, modify, and distribute the code under the terms of the license.
## Citation 
To appear in DSAA 2025: 3D Point Cloud Classification using Graph Convolutional Network with Multi-Scale Pooling\
Please find the citation information in CITATION.cff file.

## Acknowledgement 
**_This research was supported by the Polish Ministry of Science and Higher Education under project no. W87 (MNiSW/2025/DPI/73) "Computational Intelligence Methods for Image Processing in Federated Learning Systems" as part of "Supporting students to improve their competencies and skills" program._**

![Logo](logo.jpg)