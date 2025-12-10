

### **1. Introduction**
* [cite_start]**Overview:** Random Forest is an ensemble learning algorithm that combines multiple decision trees to build a robust prediction model[cite: 5]. [cite_start]It constructs numerous trees during training on random data subsets and features[cite: 6].
* [cite_start]**Mechanism:** It aggregates predictions via voting (classification) or averaging (regression) to improve generalization and reduce overfitting[cite: 7].
* [cite_start]**Goal:** The project provides hands-on experience in implementing and optimizing a Random Forest algorithm for real-world applications[cite: 8]. [cite_start]It covers the full pipeline from preprocessing to evaluation[cite: 9].
* [cite_start]**Deadline:** The submission deadline is 10/12/2025[cite: 10].

### **2. Objectives**
By completing this project, students will:
* [cite_start]Implement and optimize a Random Forest algorithm using scikit-learn[cite: 13].
* [cite_start]Apply feature engineering and selection techniques[cite: 14].
* [cite_start]Conduct systematic hyperparameter tuning[cite: 15].
* [cite_start]Evaluate performance using appropriate metrics[cite: 16].
* [cite_start]Present findings scientifically[cite: 17].

### **3. Dataset Options**
[cite_start]Students must select **ONE** of the following datasets[cite: 19]:

1.  [cite_start]**Breast Cancer Wisconsin Dataset:** Binary classification with 30 numerical features and 569 observations[cite: 20].
2.  [cite_start]**California Housing Dataset:** Housing price prediction with 8 numerical features and 20,640 observations[cite: 22, 23].
3.  [cite_start]**Credit Card Fraud Detection Dataset:** Transactions dataset (fraudulent vs. legitimate) with 31 features (28 features are PCA-transformed) and 284,807 transactions[cite: 26, 27].
4.  [cite_start]**Wine Quality Dataset:** Classification or regression dataset regarding physicochemical properties, containing 11 numerical features and 4,898 observations[cite: 29, 30].
5.  [cite_start]**Heart Disease Dataset:** Clinical features for diagnosis with 13 features and 303 observations[cite: 32, 33].
6.  [cite_start]**Forest Cover Type Dataset:** Multi-class classification with 54 features (including binary columns) and 581,012 observations[cite: 35, 36].

### **4. Technical Requirements**

#### **Data Preprocessing**
* [cite_start]Handle missing values appropriately[cite: 41].
* [cite_start]Implement feature scaling (StandardScaler or MinMaxScaler)[cite: 42].
* [cite_start]Perform feature encoding for categorical variables[cite: 43].
* [cite_start]Split data into training ($80\%$) and testing ($20\%$) sets[cite: 44].
* [cite_start]Document all preprocessing decisions[cite: 45].

#### **Random Forest Implementation**
* [cite_start]Use `RandomForestClassifier` or `RandomForestRegressor` from scikit-learn[cite: 47].
* [cite_start]Implement k-fold cross-validation where $k=5$[cite: 48].
* [cite_start]Tune the following hyperparameters and compare performance before and after tuning[cite: 49]:
    * [cite_start]`n_estimators`: range $[100, 1000]$[cite: 50].
    * [cite_start]`max_depth`: range $[5, 30]$[cite: 51].
    * [cite_start]`min_samples_split`: range $[2, 20]$[cite: 52].
    * [cite_start]`min_samples_leaf`: range $[1, 10]$[cite: 53].
    * [cite_start]`max_features`: `['auto', 'sqrt', 'log2']`[cite: 54].

#### **Visualization & Improvements**
* [cite_start]Visualize the decision tree using `plot_tree` from sklearn[cite: 55].
* [cite_start]**Performance Improvement:** Implement at least **TWO** of the following[cite: 56, 57]:
    * [cite_start]Feature selection based on importance scores[cite: 58].
    * [cite_start]Ensemble with another model (e.g., Gradient Boosting)[cite: 59].
    * [cite_start]Handle class imbalance using SMOTE or undersampling[cite: 60].
    * [cite_start]Create new features through feature engineering[cite: 61].

### **5. Deliverables**

#### **1. Code Submission (40%)**
Submit a Jupyter notebook containing:
* [cite_start]Well-documented code with clear comments[cite: 64].
* [cite_start]Preprocessing pipeline approach[cite: 65].
* [cite_start]Model implementation and optimization[cite: 66].
* [cite_start]Visualizations of results[cite: 67].

#### **2. Project Report (40%)**
Submit a report (preferably LaTeX) covering:
* [cite_start]**Introduction:** Problem statement, dataset description, and objectives [cite: 69-72].
* [cite_start]**Methodology:** Preprocessing steps, implementation details, and tuning approach [cite: 74-77].
* [cite_start]**Results:** Performance metrics, feature importance analysis, visualization, and discussion [cite: 78-82].
* [cite_start]**Conclusion:** Summary of findings, limitations, and future improvements [cite: 83-86].

#### **3. Evaluation Metrics (20%)**
* [cite_start]**For Classification:** Accuracy, Precision, Recall, F1-Score, ROC-AUC curve, and Confusion matrix [cite: 88-91].
* [cite_start]**For Regression:** $R^{2}$ Error, Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) [cite: 92-96].

### **6. Grading Weights**

| Component | Weight |
| :--- | :--- |
| Code functionality and documentation | [cite_start]15% [cite: 98] |
| Preprocessing effectiveness | [cite_start]10% [cite: 98] |
| Model implementation correctness | [cite_start]15% [cite: 98] |
| Performance optimization efforts | [cite_start]20% [cite: 98] |
| Quality of analysis and visualizations | [cite_start]20% [cite: 98] |
| Report clarity and completeness | [cite_start]20% [cite: 98] |

### **7. Submission Guidelines**
* [cite_start]Zip and submit through the course management system or by email[cite: 101].
* **Required files:**
    * [cite_start]Jupyter notebook (`.ipynb`)[cite: 103].
    * [cite_start]LaTeX report (`.tex` and `.pdf`)[cite: 104].
    * [cite_start]`Requirements.txt`[cite: 105].

