# Credit Card Fraud Detection

A machine learning project that tackles the challenge of detecting fraudulent credit card transactions in a highly imbalanced dataset (only 0.172% fraud rate). 

We progressively built and compared multiple models, from Decision Trees to XGBoost, to find the best approach for catching fraud while minimizing false alarms. The project includes advanced techniques like SMOTE oversampling, hyperparameter tuning, threshold optimization.

**Dataset:** [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 492 frauds)

## Results

| Model | F1 Score | Frauds Caught | False Alarms |
|-------|----------|---------------|--------------|
| **XGBoost** | **0.85** | **74/95** | **3** |
| Random Forest | 0.83 | 68/95 | 1 |
| Bagging | 0.79 | 63/95 | 1 |
| Decision Tree | 0.67 | 60/95 | 23 |

XGBoost performs best, catching 78% of frauds with 96% precision.

## File Structure

- `project-step-1.ipynb` - Decision Tree baseline
- `project-step-2.ipynb` - Bagging ensemble
- `project-step-3-random-forest.ipynb` - Random Forest
- `project-step-4-xgboost.ipynb` - XGBoost with SMOTE
- `project-final-comparison.ipynb` - Model comparison + threshold tuning + cost analysis

## Key Techniques

- Data preprocessing and EDA
- Decision Trees and ensemble methods
- Class weighting and SMOTE for imbalanced data
- Hyperparameter tuning with RandomizedSearchCV
- XGBoost implementation
- Threshold tuning for precision-recall optimization
- Financial cost analysis

## Setup Instructions

```bash
pip install numpy pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn jupyter kagglehub
jupyter notebook
```

Open any notebook and run. Dataset auto-downloads from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
