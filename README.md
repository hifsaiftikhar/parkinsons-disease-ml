# Parkinson's Disease Detection — Machine Learning Project

## Overview
A complete machine learning pipeline for Parkinson's disease detection 
using voice measurement data. The project covers data preprocessing, 
supervised classification, and unsupervised clustering across three 
real-world datasets from the UCI Machine Learning Repository.

## Datasets
| Dataset | Source | Type | Samples | Features |
|---|---|---|---|---|
| Parkinson's Voice Data (D1) | UCI / Local | Classification | 195 | 22 |
| Telemonitoring (D2) | UCI ML Repository | Regression | 5875 | 19 |
| Oxford Parkinson's (D3) | UCI ML Repository | Classification | 195 | 22 |

## Project Structure
| File | Description |
|---|---|
| `PREPROCESSING.ipynb` | Outlier detection, capping, normalization, train/test split |
| `SUPERVISED_LEARNING.ipynb` | 7 classification models with cross-dataset testing |
| `UNSUPERVISED_LEARNING.ipynb` | K-Means, Hierarchical, GMM clustering |
| `Day14_Parkinsons_Disease_Data.csv` | Dataset 1 (local file) |

## Notebooks — Run Order
Run notebooks in this exact order:
1. `PREPROCESSING.ipynb` — generates 6 preprocessed CSV files
2. `SUPERVISED_LEARNING.ipynb` — loads preprocessed files, trains and evaluates models
3. `UNSUPERVISED_LEARNING.ipynb` — loads preprocessed files, runs clustering

## Key Design Decisions

### Preprocessing
- Train/test split done **before** scaling to prevent data leakage
- Outlier detection using IQR method on training data only
- Outlier treatment using **Winsorization** (capping at 1st/99th percentile)
- MinMaxScaler normalization fitted on train, applied to both train and test
- Dataset 2 excluded from classification — it is a regression dataset

### Supervised Learning
- Trained on Dataset 1, tested on Dataset 3 (cross-dataset generalization)
- Dataset 1 test set used as validation to detect overfitting
- `class_weight=balanced` applied where supported to handle class imbalance
- Primary metrics: F1-Score, Recall, AUC-ROC (not accuracy — due to class imbalance)

### Unsupervised Learning
- All 3 datasets clustered without using labels
- Best k selected automatically using Silhouette Score
- GMM components selected using BIC score
- Divisive clustering properly implemented (top-down recursive splitting)
- Cluster quality measured using ARI and NMI against true labels

## Models Used

### Supervised
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting

### Unsupervised
- K-Means Clustering
- Agglomerative Hierarchical Clustering
- Divisive Hierarchical Clustering
- Gaussian Mixture Model (GMM)

## Evaluation Metrics

### Supervised
| Metric | Why Used |
|---|---|
| F1-Score | Balances precision and recall for imbalanced classes |
| Recall | Critical in medical diagnosis — missing a case is dangerous |
| AUC-ROC | Measures model discrimination ability |
| Accuracy | Reported but not primary metric due to class imbalance |

### Unsupervised
| Metric | Why Used |
|---|---|
| Silhouette Score | Measures cluster separation without labels |
| ARI | Compares clusters to true labels |
| NMI | Measures shared information between clusters and true labels |

## How to Run

### Option 1 — Google Colab (Recommended)
1. Upload `PREPROCESSING.ipynb` to Google Colab
2. Upload `Day14_Parkinsons_Disease_Data.csv` to Colab files
3. Run all cells — this generates preprocessed CSV files
4. Download the 6 generated CSV files
5. Upload `SUPERVISED_LEARNING.ipynb` to Colab
6. Upload the 6 preprocessed CSV files
7. Run all cells
8. Repeat step 5-7 for `UNSUPERVISED_LEARNING.ipynb`

### Option 2 — Jupyter Notebook (Local)
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
jupyter notebook
```
Open notebooks in order and run all cells.

## Tech Stack
- Python 3
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- scipy

## Author
**Hifsa Iftikhar**
GitHub: [@hifsaiftikhar](https://github.com/hifsaiftikhar)
