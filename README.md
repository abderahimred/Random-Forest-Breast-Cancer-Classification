# Random Forest Classification - Breast Cancer Wisconsin Dataset

A machine learning project implementing Random Forest algorithm for breast cancer diagnosis classification.

## Project Overview

This project uses the Random Forest algorithm to classify breast tumors as benign or malignant based on the Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository. The implementation covers the complete ML pipeline from data preprocessing to model evaluation and optimization.

## Dataset

- **Source:** UCI Machine Learning Repository (ID: 17)
- **Samples:** 569
- **Features:** 30 numerical features
- **Target:** Binary classification (Benign/Malignant)

## Features

- Data preprocessing with StandardScaler
- Baseline Random Forest model
- Hyperparameter tuning with GridSearchCV
- 5-fold cross-validation
- Feature selection based on importance scores
- Class imbalance handling using SMOTE
- Comprehensive evaluation metrics
- Decision tree visualization

## Project Structure

```
tp2/
├── random_forest_breast_cancer.md    # Jupyter notebook content
├── report.md                          # LaTeX report in markdown
├── context.md                         # Project requirements
├── README.md                          # This file
└── requirements.txt                   # Python dependencies
```

## Requirements

- Python 3.8 or higher
- See requirements.txt for package dependencies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/random-forest-breast-cancer.git
cd random-forest-breast-cancer
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Open the notebook in Jupyter:
```bash
jupyter notebook
```

2. Copy the code from `random_forest_breast_cancer.md` into a new notebook or convert it using jupytext:
```bash
jupytext --to notebook random_forest_breast_cancer.md
```

3. Run all cells to train and evaluate the model.

## Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Baseline | 0.9649 | 0.9535 | 0.9535 | 0.9535 | 0.9918 |
| Tuned | 0.9737 | 0.9545 | 0.9767 | 0.9655 | 0.9953 |
| Feature Selection | 0.9737 | 0.9545 | 0.9767 | 0.9655 | 0.9942 |
| SMOTE | 0.9649 | 0.9302 | 0.9767 | 0.9530 | 0.9942 |
| Combined | 0.9649 | 0.9302 | 0.9767 | 0.9530 | 0.9930 |

## Key Findings

1. Random Forest achieves over 96% accuracy on this dataset
2. Top predictive features include concave points, perimeter, and radius measurements
3. Feature selection reduces dimensionality from 30 to 15 features with minimal performance loss
4. SMOTE improves recall for the minority class (malignant tumors)

## Performance Improvements

Two optimization techniques are implemented:

1. **Feature Selection:** Uses importance scores to select the most predictive features
2. **SMOTE:** Synthetic Minority Over-sampling Technique to handle class imbalance

## License

This project is for educational purposes.

## Author

Machine Learning Course Project - December 2025
