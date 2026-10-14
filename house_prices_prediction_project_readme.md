# 🏠 Residential House Price Prediction using Ensembled Machine Learning

An end-to-end Machine Learning pipeline built to accurately predict residential housing values. The system leverages advanced feature engineering, Box-Cox skewness corrections, and an ensemble architecture combining **Regularized Linear Regressors (Lasso, ElasticNet)**, **Kernel Ridge Regression**, **Gradient Boosted Trees (XGBoost, LightGBM, GradientBoosting)**, and a **Stacked Generalization Meta-Learner**.

---

## 📌 Project Overview

Predicting property sales prices requires addressing complex challenges such as high-dimensional feature spaces, non-linear relationships, missing data patterns, and non-normally distributed targets. 

This repository implements a robust data-preprocessing and multi-tiered modeling strategy designed to mitigate overfitting and maximize generalization performance on unseen test data.

### Key Objectives
- Clean and impute missing data using domain-specific heuristics and localized neighborhood statistics.
- Perform feature selection and apply Box-Cox transformation to normalise right-skewed numerical variables.
- Design a two-layer **Stacking Regressor** combined with gradient boosting algorithms.
- Generate formatted final predictions (`submission.csv`) for model evaluation.

---

## 🏗️ Pipeline Architecture

```
                 ┌────────────────────────────────┐
                 │     Raw Housing Data (CSV)     │
                 └───────────────┬────────────────┘
                                 │
                                 ▼
                 ┌────────────────────────────────┐
                 │ Preprocessing & Imputation     │
                 │  - Outlier filtering           │
                 │  - Neighborhood median fills   │
                 │  - One-Hot / Label Encoding    │
                 └───────────────┬────────────────┘
                                 │
                                 ▼
                 ┌────────────────────────────────┐
                 │ Feature Engineering & Skewness │
                 │  - Target log1p transformation │
                 │  - TotalSF composite metric    │
                 │  - Box-Cox (λ = 0.15)          │
                 └───────────────┬────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ Stacked Model    │   │ XGBoost          │   │ LightGBM         │
│ Base: ENet, GBR, │   │ Subsample: 0.52  │   │ Num Leaves: 5    │
│ KRR              │   │ Colsample: 0.46  │   │ Feature Frac:    │
│ Meta: Lasso      │   │                  │   │ 0.2319           │
└────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
         │                      │                      │
         └──────────────────┬───┴──────────────────────┘
                            │ (Weighted Average: 70% Stack / 15% XGB / 15% LGB)
                            ▼
             ┌──────────────────────────────┐
             │ Final SalePrice Predictions  │
             │     (`submission.csv`)       │
             └──────────────────────────────┘
```

---

## 📊 Model Performance & Architecture Summary

The final model uses a weighted ensemble strategy to leverage the distinct strengths of both linear regularized models and tree-based gradient boosters:

$$\text{Final Prediction} = 0.70 \times \hat{y}_{\text{Stacked}} + 0.15 \times \hat{y}_{\text{XGBoost}} + 0.15 \times \hat{y}_{\text{LightGBM}}$$

| Model Layer | Algorithm | Key Hyperparameters / Configuration | Role in Ensemble |
| :--- | :--- | :--- | :--- |
| **Base Model** | **Lasso Regression** | $\alpha = 0.0005$, `RobustScaler()` | Regularized linear baseline / Meta-learner |
| **Base Model** | **ElasticNet** | $\alpha = 0.0005$, $l_1\text{-ratio} = 0.9$, `RobustScaler()` | L1/L2 penalty balanced feature weighting |
| **Base Model** | **Kernel Ridge (KRR)** | $\alpha = 0.6$, `polynomial` kernel, $\text{degree} = 2$ | Captures non-linear feature interactions |
| **Base Model** | **Gradient Boosting** | $n\_est = 3000$, $lr = 0.05$, `huber` loss | Sequential error correction via decision trees |
| **Meta-Learner** | **Stacked Averaged Model**| Out-Of-Fold ($K=5$) Cross-Validation | Combines base estimates to optimize generalization |
| **Boosting** | **XGBoost Regressor** | $n\_est = 2200$, $lr = 0.05$, $colsample = 0.46$ | Handles high-dimensional sparse representations |
| **Boosting** | **LightGBM Regressor**| $n\_est = 720$, $num\_leaves = 5$, $lr = 0.05$ | Leaf-wise tree growth optimization |

---

## 📈 Visualizations & Evaluation

To evaluate model predictions against actual ground-truth values during training, the pipeline generates a regression fit comparison chart.

### Actual vs. Predicted Target Values
The plot evaluates ground-truth log sale prices against predictions output by the primary stacked regressor:

```
    Predicted Log Price
          ▲
          │                                  * (Perfect Fit line: y = x)
   13.5 ──┼─────────────────────────────*───
          │                          *  *
   13.0 ──┼──────────────────────*  *
          │                   *  *  *
   12.5 ──┼────────────────*  *
          │             *  *
   12.0 ──┼──────────*  *
          │       *
   11.5 ──┼────*
          └──────┬───────┬───────┬───────┬───► Actual Log Price
               11.5    12.0    12.5    13.0   13.5
```

### Generated Output Artifacts
When executed, the script creates:
1. **`actual_vs_predicted.png`**: High-resolution scatter plot saved to disk illustrating training residuals and overall model fit quality.
2. **`submission.csv`**: Formatted output file containing target predictions formatted for Kaggle evaluation.

---

## ⚙️ Installation & Environment Setup

### Prerequisites
- **Python:** Version `3.8` or higher
- **Package Manager:** `pip` or `conda`

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/house-prices-prediction.git
cd house-prices-prediction
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
Install all required libraries via `pip`:
```bash
pip install numpy pandas scipy scikit-learn xgboost lightgbm matplotlib seaborn
```

---

## 🚀 How to Run the Project

1. **Obtain Data:** Download `train.csv` and `test.csv` from the [Kaggle House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).
2. **Place Datasets:** Place both CSV files in the project root directory (or update the directory path in `house_prices_model.py`).
3. **Execute Script:** Run the end-to-end execution script:
   ```bash
   python house_prices_model.py
   ```
4. **Inspect Output:** Once the run completes, check the root directory for `submission.csv` and `actual_vs_predicted.png`.

---

## 📁 Repository Structure

```
├── house_prices_model.py     # End-to-end data processing and modeling script
├── README.md                 # Technical project overview and documentation
├── actual_vs_predicted.png   # Generated visualization figure
└── submission.csv            # Formatted prediction output file
```