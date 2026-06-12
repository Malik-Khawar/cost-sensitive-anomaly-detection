# 🛡️ Cost-Sensitive High-Dimensional Anomaly Detection

This repository contains a standalone **Cost-Sensitive Anomaly Detection** pipeline built with **Scikit-learn, Imbalanced-learn, and Category Encoders**. It supports both synthetic data and the **real Credit Card Fraud Detection dataset** (284,807 European credit card transactions from September 2013, sourced from OpenML).

In real-world applications like financial fraud detection, statistical metrics (e.g., F1-score or PR-AUC) do not tell the whole story. Asymmetrical business costs exist:
- **False Negatives (Missed Fraud)**: Extremely expensive; costs the transaction amount plus chargeback fees.
- **False Positives (False Alarms)**: Costs customer goodwill (friction/irritation) and verification overhead.
- **True Positives (Correct Blocks)**: Requires a small manual verification cost.

This project implements a framework that optimizes machine learning decision thresholds directly for **Expected Monetary Value (EMV)** rather than default statistical classification cuts.

---

## 🚀 Key Features

1. **Custom Financial Cost Matrix**: Dynamically computes business utility based on actual transaction values and user friction costs.
2. **Real Dataset Support**: Download and evaluate on the real [Credit Card Fraud Detection dataset](https://www.openml.org/d/1597) with 284K transactions and 0.17% fraud rate using the `--real-data` flag.
3. **High-Cardinality Target Encoding**: Handles categorical identifiers (`merchant_id`, `device_id`) using regularized target encoding to prevent overfitting (synthetic data mode).
4. **Advanced Class Imbalance Handling**: Compares pipeline architectures:
   - Class-weighted Logistic Regression.
   - SMOTE (Synthetic Minority Over-sampling Technique) combined with Random Forest.
   - Cost-Sensitive Random Forest with class-priority weighting.
5. **Decision Threshold Tuning for Savings**: Computes optimal decision thresholds to maximize net financial savings relative to a "do-nothing" baseline.

---

## 🔬 Mathematical Formulation

### 1. The Financial Utility Matrix
For a transaction with value $A$, we define the utility $U(y, \hat{y})$ for actual label $y \in \{0, 1\}$ and prediction $\hat{y} \in \{0, 1\}$:

- **True Negative (TN)**: $U(0, 0) = 0$ (Approved clean transaction).
- **False Positive (FP)**: $U(0, 1) = -C_{\text{irritate}}$ (Clean transaction blocked; set to $-\$25$ representing support and user irritation costs).
- **True Positive (TP)**: $U(1, 1) = -C_{\text{verify}}$ (Fraud correctly blocked; set to $-\$10$ representing verification cost).
- **False Negative (FN)**: $U(1, 0) = -(A + C_{\text{chargeback}})$ (Fraud transaction approved; loss is full transaction amount $A$ plus a chargeback fee $C_{\text{chargeback}} = \$15$).

### 2. Expected Net Savings
Relative to the baseline where all transactions are approved (meaning all fraud results in FN losses), the net savings of a model is:
$$\text{Net Savings} = \text{Utility}_{\text{Model}} - \text{Utility}_{\text{Baseline}}$$
$$\text{Utility}_{\text{Baseline}} = \sum_{i \in \text{Fraud}} -(A_i + C_{\text{chargeback}})$$

---

## 📊 Verification & Results

Evaluating on a synthetic dataset of 10,000 transactions (0.53% train fraud rate) yields the following results on the test set (3,000 transactions; $13$ fraud cases totaling $\$22,817.30$ in potential loss):

| Model Name | ROC-AUC | PR-AUC | Optimal Threshold | Optimal Savings | Optimal Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Balanced Logistic Regression** | `0.9979` | `0.8801` | `0.6262` | **\$21,912.31** | `92.31%` |
| **SMOTE + Random Forest** | `0.9990` | `0.8433` | `0.3031` | **\$22,162.31** | `92.31%` |
| **Cost-Sensitive Random Forest** | `0.9993` | `0.8888` | `0.0607` | **\$22,187.30** | `100.00%` |

*The Cost-Sensitive Random Forest saves **97.24%** of the potential fraud loss by blocking all 13 fraud transactions at its optimal threshold.*

### Saved Outputs
- **Savings Curve (`results/savings_threshold_curve.png`)**: Charts net financial savings as a function of the probability threshold, showing how statistical cuts (like 0.50) are financially sub-optimal.
- **Model Comparison (`results/model_savings_comparison.png`)**: Compares standard vs optimized savings for all three models.

---

## 📁 Repository Structure

```text
├── src/
│   ├── data.py         # Synthetic fraud generator + real dataset loader (OpenML)
│   ├── features.py     # Preprocessing pipeline (target encoder, scaling, PCA)
│   ├── cost_matrix.py  # Asymmetric financial utility and savings equations
│   ├── models.py       # Scikit-learn and Imbalanced-learn model pipelines
│   ├── evaluate.py     # Evaluation functions for cost and standard metrics
│   └── visualize.py    # Plots for savings-threshold curves and model comparison
├── results/            # Performance visualization charts
├── main.py             # Pipeline entrypoint script
├── pyproject.toml      # Package config and project meta
└── requirements.txt    # Standard pip requirements file
```

---

## ⚙️ Installation & Usage

### Setup Environment
Install dependencies using `pip`:
```bash
pip install -r requirements.txt
```

### Run Evaluation

**Synthetic data** (default):
```bash
python main.py --samples 10000
```

**Real Credit Card Fraud dataset** (downloads from OpenML on first run):
```bash
python main.py --real-data
```

The real dataset contains 284,807 transactions with 28 PCA-transformed features (V1-V28), plus Time and Amount. When `--real-data` is used, the full dataset is processed (sorted chronologically).
