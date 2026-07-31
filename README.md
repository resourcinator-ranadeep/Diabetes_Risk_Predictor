<div align="center">

# 🩺 Diabetes Risk Predictor

**An End-to-End Machine Learning Pipeline for Predicting Diabetes Risk Across Both Genders**

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

**Model Accuracy: 96%** &nbsp;|&nbsp; **Diabetic Recall: 79%** &nbsp;|&nbsp; **Memory Footprint: < 150 MB**

</div>

---

## 📑 Table of Contents

- [📌 Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
- [🗄️ Dataset](#️-dataset)
- [🖥️ Hardware Constraints & Optimization](#️-hardware-constraints--optimization)
- [🧪 Algorithmic Iterations](#-algorithmic-iterations)
- [📊 Model Performance](#-model-performance)
- [🚀 Getting Started](#-getting-started)
- [💻 Usage](#-usage)
- [📁 Project Structure](#-project-structure)
- [📜 License](#-license)
- [🙏 Acknowledgements](#-acknowledgements)

---

## 📌 Project Overview

This project builds a **binary classification model** that predicts whether a patient is at risk of diabetes, based on clinical and lifestyle indicators such as age, BMI, blood glucose level, HbA1c level, hypertension, heart disease, gender, and smoking history.

The workflow is specifically engineered to handle a **severely imbalanced dataset** (≈91.5% healthy vs. ≈8.5% diabetic patients) while running comfortably on **resource-constrained hardware** — an Intel i3-4160 CPU with just **4 GB DDR3 RAM**.

---

## ✨ Key Features

- 🔄 **Complete ML Pipeline** — data ingestion → preprocessing → model training → evaluation
- ⚖️ **Custom Class-Weighting** — clinically balanced `1:5` misclassification penalty ratio
- 💾 **Low-RAM Optimization** — memory-efficient dtype downcasting keeps execution under **150 MB**
- ⚡ **CPU-Optimized** — parallelized Random Forest training via `n_jobs=-1`
- 🌐 **Cross-Gender Support** — one-hot encoded categorical features for both `gender` and `smoking_history`
- 📊 **Detailed Reporting** — precision, recall, and F1-score breakdown per class

---

## 🗄️ Dataset

| Attribute | Details |
|---|---|
| **Source** | [Kaggle — Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) |
| **Author** | *iammustafatz* |
| **Records** | 100,000 patient records |
| **Target Variable** | `diabetes` (binary: 0 = Non-Diabetic, 1 = Diabetic) |
| **Class Balance** | ⚠️ Highly imbalanced — ~91.5% healthy / ~8.5% diabetic |
| **Features** | `gender`, `age`, `hypertension`, `heart_disease`, `smoking_history`, `bmi`, `HbA1c_level`, `blood_glucose_level` |

---

## 🖥️ Hardware Constraints & Optimization

This pipeline was designed to deliver strong performance on modest hardware. The following strategies keep it efficient:

| Constraint | Optimization Strategy |
|---|---|
| **4 GB RAM limit** | Script runs as a plain `.py` file (no heavy browser/IDE overhead) |
| **Large dataset** | Columns downcast to space-efficient `float32`, `int8`, and `category` dtypes |
| **Algorithmic complexity** | Random Forest capped at `n_estimators=50` & `max_depth=10` |
| **Execution footprint** | Total memory usage maintained **under 150 MB** |
| **Runtime speed** | `n_jobs=-1` utilizes all available CPU cores |

---

## 🧪 Algorithmic Iterations

The severe class imbalance posed a significant challenge. Three major iterations were explored to find the clinical "sweet spot":

| Iteration | Approach | Precision | Recall | Key Takeaway |
|:---:|---|:---:|:---:|---|
| **1** | Baseline **Logistic Regression** | **0.87** | **0.60** | ⚠️ Too conservative — missed **40%** of diabetic patients |
| **2** | Auto-Balanced **Random Forest** (`class_weight='balanced'`) | **0.44** | **0.92** | ⚠️ Over-corrected — excessive false alarms cratered precision |
| **3** ✅ | **Random Forest** with custom `1:5` class weights | **0.75** | **0.79** | ✔️ **Balanced clinical stability** — the current champion |

### 🏆 The Winning Tweak

Instead of the aggressive auto-balancing (`balanced`) or ignoring imbalance entirely, a **manual misclassification penalty** was applied:

```python
custom_weights = {0: 1, 1: 5}   # A diabetic case is weighted 5× a healthy case
```

This strikes a **clinically meaningful equilibrium**: it safely catches **79% of diabetic individuals** while keeping **75% precision** — minimizing both missed diagnoses *and* false alarms.

---

## 📊 Model Performance

Classification report from the final iteration on the **20,000-record test split** (80/20 split, `random_state=42`):

```
              precision    recall  f1-score   support

           0       0.98      0.98      0.98     18292
           1       0.75      0.79      0.77      1708

    accuracy                           0.96     20000
   macro avg       0.86      0.88      0.87     20000
weighted avg       0.96      0.96      0.96     20000
```

| Metric | Value |
|---|---|
| 🎯 **Overall Accuracy** | **96%** |
| 🟢 **Healthy Precision / Recall** | 98% / 98% |
| 🔴 **Diabetic Precision / Recall** | 75% / 79% |
| 🟣 **Diabetic F1-Score** | 0.77 |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- `pip` package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/RANADEEP/Diabetes_Risk_Predictor.git
cd Diabetes_Risk_Predictor

# Install dependencies
pip install pandas scikit-learn
```

> 📥 Ensure the dataset file `diabetes_prediction_dataset.csv` is present in the project root (download it from [Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) if needed).

---

## 💻 Usage

Run the complete pipeline (training + evaluation) with:

```bash
python predict_diabetes.py
```

This will:
1. **Ingest** the CSV with memory-optimized dtypes
2. **Preprocess** features (scaling numerics, one-hot encoding categoricals)
3. **Split** the data 80/20 for training/testing
4. **Train** a Random Forest with the custom `1:5` class weights
5. **Output** the full classification report

---

## 📁 Project Structure

```
Diabetes_Risk_Predictor/
├── predict_diabetes.py                 # Main ML pipeline script
├── DiabetesPred.ipynb                  # Jupyter notebook experimentation
├── diabetes_prediction_dataset.csv     # Kaggle dataset (100k records)
├── diabetes_prediction_dataset.csv.zip # Compressed dataset archive
├── Diabetes_Risk_Pred_Proposal.docx    # Project proposal document
├── Diabetes_Risk_Pred_Proposal.pdf     # Project proposal (PDF)
├── progress.txt                        # Iteration & progress log
├── LICENSE                             # MIT License
└── README.md                           # You are here 👋
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE). See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- 🙌 **Dataset Author** — [iammustafatz](https://www.kaggle.com/iammustafatz) for the *Diabetes Prediction Dataset*
- 🛠️ **Libraries** — [Pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/)
- 💡 **Inspiration** — Building clinically responsible ML that works on everyday hardware

---

<div align="center">

**Made with ❤️ for accessible, hardware-friendly healthcare AI**

</div>
