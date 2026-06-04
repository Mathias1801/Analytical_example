# Kiva Microfinance — Data Exploration & Applied Machine Learning

An end-to-end data science project on the [Kiva](https://www.kiva.org/) crowdfunding dataset, covering the full workflow from data cleaning and exploratory analysis through unsupervised learning to supervised regression with model explainability.

The guiding question: **how do borrower and loan characteristics relate to the loan amount disbursed, and how well can that amount be predicted?**

## Overview

Kiva is a nonprofit that lets individuals lend small amounts to low-income entrepreneurs and students across 77 countries. Using a public dataset of Kiva loan activity, this notebook quantifies how features such as sector, country, currency, lender count, loan term, and repayment interval relate to the loan amount, and benchmarks several models for predicting it.

## What's inside

| Stage | Techniques |
|-------|-----------|
| **Data preparation** | Multi-file loading, column selection, missing-value handling, IQR-based outlier removal |
| **Exploratory analysis** | Descriptive statistics, distribution and box plots (Seaborn), group-by aggregation, correlation heatmap |
| **Unsupervised learning** | K-means clustering (with elbow method), agglomerative hierarchical clustering & dendrograms, an SVD-based recommendation prototype (Truncated SVD + cosine similarity) |
| **Supervised learning** | Linear Regression, Random Forest, and XGBoost regressors with one-hot encoding and feature scaling |
| **Explainability** | SHAP values (summary, beeswarm, and dependence plots) to interpret the XGBoost model |

## Results

Loan amount was predicted from the borrower/loan features, with tree-based models clearly outperforming the linear baseline:

| Model | R² | Error |
|-------|------|-------|
| Linear Regression | 0.72 | MAE ≈ 159 |
| Random Forest | 0.86 | MSE ≈ 29,060 |
| XGBoost | — | Test RMSE ≈ 616 |

SHAP analysis was then used to identify which features drive the XGBoost predictions, adding interpretability on top of the predictive performance.

## Tech stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `kmodes` · `imbalanced-learn` · `SHAP` · `SciPy` · `Matplotlib` · `Seaborn`

## Dataset

The data comes from Kaggle's **[Data Science for Good: Kiva Crowdfunding](https://www.kaggle.com/datasets/kiva/data-science-for-good-kiva-crowdfunding)** dataset.

The notebook expects the CSV files in a local `data/` folder:

```
data/
├── kiva_loans_part_0.csv
├── kiva_loans_part_1.csv
├── kiva_loans_part_2.csv
├── kiva_mpi_region_locations.csv
├── loan_theme_ids.csv
└── loan_themes_by_region.csv
```

> The original `kiva_loans.csv` from Kaggle is a single large file. This project splits it into three parts (`kiva_loans_part_0/1/2.csv`). If you download the raw file, either split it into three parts or adjust the loading cell to read `kiva_loans.csv` directly.

## Running it

```bash
pip install pandas numpy scikit-learn xgboost kmodes imbalanced-learn shap matplotlib seaborn
jupyter notebook KIVA_microfinance_ml.ipynb
```

Then place the dataset files in `data/` and run the cells top to bottom. The notebook ships with its cell outputs intact, so the plots and results are visible on GitHub without running anything.
