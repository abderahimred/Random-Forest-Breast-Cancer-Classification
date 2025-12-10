# LaTeX Report

Copy the following LaTeX code into a `.tex` file:

```latex
\documentclass[12pt,a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage{float}

\geometry{margin=2.5cm}

\begin{document}

% Title Page
\begin{titlepage}
    \centering
    \vspace*{2cm}
    
    {\Large \textbf{Data Science Course}}\\[0.5cm]
    
    \vspace{1.5cm}
    
    {\Huge \textbf{Random Forest Classification}}\\[0.8cm]
    {\Huge \textbf{for Breast Cancer Diagnosis}}\\[1.5cm]
    
    \vspace{2cm}
    
    {\Large Authors:}\\[0.5cm]
    {\large Author Name 1}\\[0.3cm]
    {\large Author Name 2}\\[1cm]
    
    \vspace{2cm}
    
    {\large December 2025}\\[1.5cm]
    
    \vspace{2cm}
    
    \rule{0.6\textwidth}{0.4pt}\\[0.5cm]
    {\normalsize GitHub Repository:}\\[0.3cm]
    \url{https://github.com/username/repository-name}\\[0.5cm]
    \rule{0.6\textwidth}{0.4pt}
    
    \vfill
    
\end{titlepage}

\section{Introduction}

\subsection{Problem Statement}

Breast cancer is one of the most common types of cancer affecting women worldwide. Early and accurate diagnosis is crucial for effective treatment and improved survival rates. The challenge lies in correctly classifying tumors as benign or malignant based on various cell characteristics extracted from medical imaging.

This project aims to develop a robust classification model using the Random Forest algorithm to distinguish between benign and malignant breast tumors. The model will be trained on digitized images of fine needle aspirate (FNA) of breast masses, where various features of cell nuclei are computed.

\subsection{Dataset Description}

The Breast Cancer Wisconsin (Diagnostic) dataset is obtained from the UCI Machine Learning Repository. It contains 569 samples with 30 numerical features computed from cell nuclei images. The features describe characteristics such as:

\begin{itemize}
    \item Radius, texture and perimeter of cell nuclei
    \item Area and smoothness measurements
    \item Compactness and concavity values
    \item Symmetry and fractal dimension
\end{itemize}

Each sample is labeled as either Malignant (M) or Benign (B). The dataset shows a slight class imbalance with approximately 63\% benign and 37\% malignant cases. There are no missing values in the dataset, which simplifies the preprocessing phase.

\subsection{Objectives}

The main objectives of this project are:

\begin{enumerate}
    \item Implement a Random Forest classifier using scikit-learn for binary classification
    \item Apply appropriate data preprocessing techniques including feature scaling
    \item Perform systematic hyperparameter tuning using cross-validation
    \item Improve model performance through feature selection and class imbalance handling
    \item Evaluate the model using standard classification metrics
\end{enumerate}

\section{Methodology}

\subsection{Preprocessing Steps}

The preprocessing pipeline consists of several steps to prepare the data for model training.

\textbf{Target Encoding:} The diagnosis labels are converted from categorical (M/B) to numerical format where Malignant is encoded as 1 and Benign as 0. This allows the classifier to work with numerical targets.

\textbf{Feature Scaling:} StandardScaler is applied to normalize all 30 features. This transformation ensures that each feature has a mean of 0 and standard deviation of 1. Scaling is important for Random Forest when comparing feature importances.

\textbf{Data Splitting:} The dataset is split into training (80\%) and testing (20\%) sets using stratified sampling. Stratification ensures that both sets maintain the same class distribution as the original dataset. A random seed is fixed for reproducibility.

\textbf{Missing Values:} A check for missing values confirms that the dataset is complete. No imputation is required for this dataset.

\subsection{Implementation Details}

The Random Forest classifier is implemented using scikit-learn library. Five different model configurations are trained and evaluated:

\begin{enumerate}
    \item \textbf{Baseline Model:} Random Forest with default hyperparameters (100 trees, no depth limit)
    \item \textbf{Tuned Model:} Random Forest optimized through GridSearchCV
    \item \textbf{Feature Selection Model:} Tuned model using only top 15 features based on importance
    \item \textbf{SMOTE Model:} Tuned model trained on SMOTE-balanced data
    \item \textbf{Combined Model:} Feature selection combined with SMOTE oversampling
\end{enumerate}

\textbf{Cross-Validation:} 5-fold cross-validation is used throughout the project. This technique divides the training data into 5 equal parts, trains on 4 parts and validates on the remaining part, repeating this process 5 times.

\subsection{Tuning Approach}

Hyperparameter tuning is performed using GridSearchCV with the following parameter ranges:

\begin{itemize}
    \item \textbf{n\_estimators:} Number of trees in the forest, tested with values 100, 300, and 500
    \item \textbf{max\_depth:} Maximum depth of each tree, tested with values 10, 20, and 30
    \item \textbf{min\_samples\_split:} Minimum samples required to split a node, tested with 2 and 10
    \item \textbf{min\_samples\_leaf:} Minimum samples required at leaf nodes, tested with 1 and 4
    \item \textbf{max\_features:} Features considered at each split, tested with sqrt and log2
\end{itemize}

The grid search evaluates all parameter combinations using 5-fold cross-validation and selects the configuration that maximizes accuracy.

\section{Results}

\subsection{Performance Metrics}

All five models are evaluated using classification metrics. Table \ref{tab:comparison} presents a comprehensive comparison of all model variants.

\begin{table}[H]
\centering
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{ROC-AUC} \\
\midrule
Baseline & 0.9649 & 0.9535 & 0.9535 & 0.9535 & 0.9918 \\
Tuned & 0.9737 & 0.9545 & 0.9767 & 0.9655 & 0.9953 \\
Feature Selection & 0.9737 & 0.9545 & 0.9767 & 0.9655 & 0.9942 \\
SMOTE & 0.9649 & 0.9302 & 0.9767 & 0.9530 & 0.9942 \\
Combined (FS + SMOTE) & 0.9649 & 0.9302 & 0.9767 & 0.9530 & 0.9930 \\
\bottomrule
\end{tabular}
\caption{Performance comparison of all Random Forest model variants}
\label{tab:comparison}
\end{table}

\subsection{Model Comparison Analysis}

\textbf{Baseline vs Tuned Model:}

The hyperparameter tuning improves accuracy from 0.9649 to 0.9737. The most significant improvement is observed in recall, which increases from 0.9535 to 0.9767. This means the tuned model correctly identifies more malignant cases. The ROC-AUC also improves from 0.9918 to 0.9953.

\textbf{Effect of Feature Selection:}

The feature selection model achieves the same accuracy (0.9737) and recall (0.9767) as the tuned model while using only 15 out of 30 features. This demonstrates that half of the original features are redundant. The slight decrease in ROC-AUC (0.9942 vs 0.9953) is negligible. Feature selection offers benefits in terms of:
\begin{itemize}
    \item Reduced model complexity
    \item Faster prediction time
    \item Improved interpretability
\end{itemize}

\textbf{Effect of SMOTE:}

SMOTE oversampling balances the training set by generating synthetic samples for the minority class (malignant). The SMOTE model achieves high recall (0.9767) but lower precision (0.9302) compared to the tuned model (0.9545). This tradeoff is expected because:
\begin{itemize}
    \item SMOTE increases sensitivity to the minority class
    \item More aggressive prediction of malignant cases leads to more false positives
    \item In medical diagnosis, higher recall is often preferred over precision
\end{itemize}

\textbf{Combined Approach:}

The combined model (Feature Selection + SMOTE) shows similar performance to the SMOTE-only model. It maintains high recall (0.9767) while using fewer features. This approach is suitable when both dimensionality reduction and class balance are important.

\subsection{Best Model Selection}

Table \ref{tab:best} summarizes which model performs best for each metric.

\begin{table}[H]
\centering
\begin{tabular}{ll}
\toprule
\textbf{Metric} & \textbf{Best Model} \\
\midrule
Accuracy & Tuned, Feature Selection (0.9737) \\
Precision & Tuned, Feature Selection (0.9545) \\
Recall & Tuned, Feature Selection, SMOTE, Combined (0.9767) \\
F1-Score & Tuned, Feature Selection (0.9655) \\
ROC-AUC & Tuned (0.9953) \\
\bottomrule
\end{tabular}
\caption{Best performing model for each evaluation metric}
\label{tab:best}
\end{table}

The \textbf{Tuned model} and \textbf{Feature Selection model} show the best overall performance. For practical applications, the Feature Selection model is recommended since it achieves comparable results with half the features.

\subsection{Feature Importance Analysis}

Feature importance scores from the Random Forest model reveal which characteristics are most useful for classification. The top 10 most important features are:

\begin{enumerate}
    \item worst concave points
    \item worst perimeter
    \item worst radius
    \item mean concave points
    \item worst area
    \item mean perimeter
    \item mean radius
    \item mean area
    \item worst concavity
    \item mean concavity
\end{enumerate}

Features related to "worst" measurements (computed from the three largest cell nuclei) and concavity characteristics appear to be most discriminative.

\subsection{Visualization}

Several visualizations are generated to understand model behavior:

\textbf{Decision Tree Plot:} A single tree from the Random Forest is visualized using sklearn's plot\_tree function. The visualization shows the split conditions, class distributions at each node, and the decision path.

\textbf{Confusion Matrix:} The confusion matrix for the tuned model shows:
\begin{itemize}
    \item True Negatives (correctly identified benign): 70
    \item True Positives (correctly identified malignant): 42
    \item False Positives (benign predicted as malignant): 2
    \item False Negatives (malignant predicted as benign): 1
\end{itemize}

\textbf{ROC Curve:} All models achieve AUC scores above 0.99, indicating excellent discrimination ability. The curves are nearly identical, confirming that all variants perform at a high level.

\subsection{Discussion}

The Random Forest classifier demonstrates strong performance on the breast cancer classification task across all configurations. Key observations:

\begin{enumerate}
    \item All models achieve accuracy above 96\%, indicating the dataset is well-suited for Random Forest classification
    \item Hyperparameter tuning provides modest but consistent improvements
    \item Feature selection offers practical benefits without sacrificing performance
    \item SMOTE shifts the precision-recall tradeoff toward higher recall
    \item The choice between models depends on the specific use case requirements
\end{enumerate}

For medical diagnosis where missing a cancer case is critical, models with high recall (SMOTE, Combined) may be preferred. For balanced overall performance, the Tuned or Feature Selection models are recommended.

\section{Conclusion}

\subsection{Summary of Findings}

This project successfully implements and compares five Random Forest classifier variants for breast cancer diagnosis. The key findings are:

\begin{enumerate}
    \item The Random Forest algorithm achieves high classification accuracy above 96\% on this dataset
    \item Hyperparameter tuning through grid search improves accuracy by approximately 1\%
    \item Feature importance analysis identifies concavity and size measurements as most predictive
    \item Feature selection reduces the model to 15 features without significant performance loss
    \item SMOTE oversampling improves recall at the cost of slightly lower precision
    \item The combined approach provides a balanced solution for reduced complexity and class balance
\end{enumerate}

\subsection{Limitations}

Several limitations should be considered:

\begin{enumerate}
    \item The dataset is relatively small with only 569 samples
    \item The features are pre-computed from images
    \item The grid search covers a limited parameter space due to computational constraints
    \item The model is evaluated on a single train-test split
\end{enumerate}

\subsection{Future Improvements}

Potential directions for future work include:

\begin{enumerate}
    \item Combining Random Forest with other algorithms like Gradient Boosting
    \item Creating interaction features or domain specific transformations
    \item Applying neural network approaches to raw image data
    \item Testing the model on data from different hospitals or populations
    \item Adjusting the classification threshold based on clinical cost considerations
\end{enumerate}

\end{document}
```
