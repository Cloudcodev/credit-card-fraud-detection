# Credit Card Fraud Detection

A machine learning project that identifies fraudulent credit card transactions. The main challenge is class imbalance: fraud is extremely rare, so a model that labels every transaction as legitimate would still score above 99.8% accuracy while catching no fraud at all.

## Problem

Banks and payment processors need to flag fraudulent transactions quickly without blocking too many legitimate purchases. This project builds a classifier for that task and evaluates it with metrics that reflect the real goal: precision (how many flagged transactions are actually fraud) and recall (how many of the fraud cases are caught).

## Dataset

- **Source:** Credit Card Fraud Detection, published by the ULB Machine Learning Group on Kaggle
- **Size:** 284,807 transactions, made by European cardholders over two days in September 2013
- **Fraud cases:** 492 (about 0.17%)
- **Features:** 28 anonymized features (V1 to V28), produced by a PCA transformation to protect cardholder privacy, plus `Time` (seconds since the first transaction) and `Amount` (transaction value)
- **Target:** `Class`, where 0 means legitimate and 1 means fraud
- **License:** Open Database License (ODbL), as listed on the Kaggle dataset page

The raw data is not included in this repository. Download `creditcard.csv` from Kaggle and place it in `data/raw/`.

## Key finding

Only 0.17% of transactions are fraud. Any model must be judged on precision, recall, and the precision-recall curve, not accuracy.

## Exploratory findings

**Transaction amounts by class:**

| Class | Median amount | Mean amount |
|---|---|---|
| 0 (legitimate) | 22.00 | 88.29 |
| 1 (fraud) | 9.25 | 122.21 |

Fraud transactions have a higher mean but a lower median than legitimate ones. A few large fraud transactions raise the average, while most fraud cases involve small amounts. This is an observation from the data, not yet a tested conclusion. The modeling step will check whether `Amount` is useful for prediction.

The class balance chart is saved in `reports/figures/class_balance.png`, and the full analysis is in `notebooks/01_exploration.ipynb`.

## Project structure

```
credit-card-fraud-detection/
├── data/
│   ├── raw/            # original dataset (not tracked by Git)
│   └── processed/      # cleaned data produced by the code
├── notebooks/          # exploration and analysis
├── src/                # reusable Python code
│   └── data_loading.py
├── models/             # saved trained models
├── reports/
│   └── figures/        # charts for the README
├── requirements.txt
└── README.md
```

## Status

- [x] Project structure and environment
- [x] Data loading and class balance check
- [x] Exploratory data analysis
- [ ] Baseline model
- [ ] Final model and evaluation
- [ ] Results and demo

