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

\subsection{Theoretical Background}

\subsubsection{Random Forest Algorithm}

Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training. Each tree is built using a random subset of the training data (bootstrap sampling) and a random subset of features at each split. The final prediction is made by aggregating the predictions of all individual trees through majority voting for classification or averaging for regression.

The key advantages of Random Forest include:
\begin{itemize}
    \item Resistance to overfitting due to averaging multiple trees
    \item Ability to handle high dimensional data with many features
    \item Built-in feature importance estimation
    \item Robustness to noise and outliers in the data
\end{itemize}

\subsubsection{Evaluation Metrics}

The following metrics are used to evaluate model performance:

\textbf{Accuracy:} The proportion of correct predictions among all predictions made. It is calculated as:
\begin{equation}
    Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\end{equation}

\textbf{Precision:} The proportion of true positive predictions among all positive predictions. It measures how many of the predicted positives are actually positive:
\begin{equation}
    Precision = \frac{TP}{TP + FP}
\end{equation}

\textbf{Recall (Sensitivity):} The proportion of actual positives that are correctly identified. It measures how many of the actual positives are captured:
\begin{equation}
    Recall = \frac{TP}{TP + FN}
\end{equation}

\textbf{F1-Score:} The harmonic mean of precision and recall, providing a balanced measure when both metrics are important:
\begin{equation}
    F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\end{equation}

\textbf{ROC-AUC:} The Area Under the Receiver Operating Characteristic Curve. It measures the model's ability to distinguish between classes across all classification thresholds. A value of 1.0 indicates perfect classification, while 0.5 indicates random guessing.

\textbf{Confusion Matrix:} A table that shows the counts of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN). It provides a detailed breakdown of prediction outcomes.

\subsubsection{SMOTE Algorithm}

Synthetic Minority Over-sampling Technique (SMOTE) is a method for handling class imbalance. Instead of simply duplicating minority class samples, SMOTE creates synthetic examples by interpolating between existing minority samples. For each minority sample, it finds its k nearest neighbors and creates new samples along the line segments connecting them.

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

Several visualizations are generated to understand model behavior and performance.

\subsubsection{Class Distribution}

Figure \ref{fig:class_dist} shows the distribution of diagnosis classes in the dataset. The bar chart displays the count of benign and malignant samples, highlighting the class imbalance present in the data. Approximately 63\% of samples are benign while 37\% are malignant.

\begin{figure}[H]
    \centering
    % INSERT: Bar chart showing Benign vs Malignant class counts
    % \includegraphics[width=0.6\textwidth]{class_distribution.png}
    \caption{Distribution of diagnosis classes in the dataset}
    \label{fig:class_dist}
\end{figure}

\subsubsection{Feature Importance}

Figure \ref{fig:feature_imp} displays the feature importance scores computed by the Random Forest model. The horizontal bar chart ranks all 30 features by their contribution to the classification decision. Features related to concave points, perimeter, and radius measurements show the highest importance values.

\begin{figure}[H]
    \centering
    % INSERT: Horizontal bar chart of feature importances (all 30 features ranked)
    % \includegraphics[width=0.8\textwidth]{feature_importance.png}
    \caption{Feature importance scores from the Random Forest classifier}
    \label{fig:feature_imp}
\end{figure}

\subsubsection{Confusion Matrices}

Figure \ref{fig:confusion} shows the confusion matrices for all five model variants. Each heatmap displays the counts of true positives, true negatives, false positives, and false negatives. The tuned model shows:
\begin{itemize}
    \item True Negatives (correctly identified benign): 70
    \item True Positives (correctly identified malignant): 42
    \item False Positives (benign predicted as malignant): 2
    \item False Negatives (malignant predicted as benign): 1
\end{itemize}

\begin{figure}[H]
    \centering
    % INSERT: Grid of 5 confusion matrix heatmaps (2x3 layout, one for each model)
    % \includegraphics[width=0.9\textwidth]{confusion_matrices.png}
    \caption{Confusion matrices for all model variants}
    \label{fig:confusion}
\end{figure}

\subsubsection{ROC Curves}

Figure \ref{fig:roc} presents the Receiver Operating Characteristic curves for all five models. All models achieve AUC scores above 0.99, indicating excellent discrimination ability. The curves are nearly overlapping, confirming that all variants perform at a similarly high level.

\begin{figure}[H]
    \centering
    % INSERT: ROC curves plot with all 5 models overlaid, including diagonal reference line
    % \includegraphics[width=0.7\textwidth]{roc_curves.png}
    \caption{ROC curves comparison for all model variants}
    \label{fig:roc}
\end{figure}

\subsubsection{Model Performance Comparison}

Figure \ref{fig:comparison} shows a grouped bar chart comparing all five models across the evaluation metrics. This visualization makes it easy to identify which models excel in specific metrics and the overall performance differences.

\begin{figure}[H]
    \centering
    % INSERT: Grouped bar chart with 5 model groups, each having 5 bars for metrics
    % \includegraphics[width=0.85\textwidth]{model_comparison.png}
    \caption{Performance comparison across all models and metrics}
    \label{fig:comparison}
\end{figure}

\subsubsection{Decision Tree Visualization}

Figure \ref{fig:tree} displays a single decision tree from the Random Forest ensemble. The tree is limited to depth 3 for readability. Each node shows the split condition, the number of samples, and the class distribution. The color intensity indicates the dominant class at each node.

\begin{figure}[H]
    \centering
    % INSERT: Decision tree plot from sklearn's plot_tree (depth limited to 3-4)
    % \includegraphics[width=0.95\textwidth]{decision_tree.png}
    \caption{Visualization of a single decision tree from the Random Forest}
    \label{fig:tree}
\end{figure}

\subsubsection{SMOTE Effect}

Figure \ref{fig:smote} shows the class distribution before and after applying SMOTE. The side-by-side bar charts demonstrate how SMOTE balances the training set by generating synthetic minority class samples.

\begin{figure}[H]
    \centering
    % INSERT: Two bar charts side by side - before SMOTE vs after SMOTE class distribution
    % \includegraphics[width=0.7\textwidth]{smote_effect.png}
    \caption{Class distribution before and after SMOTE oversampling}
    \label{fig:smote}
\end{figure}

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
