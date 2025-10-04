# Task 4: Loan Approval Prediction

## Description
- **Objective:** Predict whether a loan application will be approved.
- **Dataset:** Loan Approval Prediction Dataset (Kaggle recommended)
- **Tasks:**
  - Handle missing values and encode categorical features.
  - Train a classification model and evaluate performance on imbalanced data.
  - Focus on **precision, recall, and F1-score**.
- **Tools & Libraries:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn.
- **Covered Topics:** Binary classification, handling imbalanced data.
- **Bonus:** Use SMOTE or other techniques for class imbalance; try Logistic Regression vs Decision Tree.

---

## Exploratory Data Analysis (EDA)

### Dataset Overview
- Dataset contains both numerical and categorical features, including `income_annum`, `loan_amount`, `cibil_score`, and asset values.
- There are missing values in some columns and a few duplicates, which were cleaned.
- Summary statistics show skewed distributions for financial attributes.

### Target Distribution
- The dataset is **imbalanced**:
  - Approved: 62.22%
  - Rejected: 37.78%
- Imbalance needs to be addressed to prevent model bias.

### Numerical Features
- **CIBIL Score:** Left-skewed; most applicants have high scores.
- **Asset Values & Loan Amount:** Right-skewed; few applicants have extremely high values.
- Outlier removal was applied using the IQR method to reduce distortion in model training.

### Categorical Features vs Target
- **Education:** Approval rates almost identical for Graduates and Non-Graduates (~62%).
- **Self-Employed:** Approval rates similar for self-employed and non-self-employed (~62%).
- **Number of Dependents:** Higher number of dependents slightly reduces approval chances (0 dependents → 64.19% approval, 5 dependents → 60.33%).
- **Insight:** Loan approval is largely unaffected by education or employment type, but slightly influenced by dependents.

### Numerical Features vs Target
- **Income:** Approved applicants have slightly higher median income.
- **Loan Amount:** Approved loans tend to have smaller amounts.
- **Loan Term:** Approved loans tend to have shorter terms.
- **CIBIL Score:** Strongest predictor; median ~700 for Approved vs ~500 for Rejected.
- **Asset Values:** Surprisingly, very high asset values show a mild negative correlation with approval.
- **Insight:** CIBIL score is critical, loan amount and term matter moderately, assets and demographics are less influential.

---

## Model Insights

### Random Forest Classifier
- **Performance:**
  - Accuracy: 98.2%
  - Precision & Recall: High for both Approved and Rejected classes.
  - F1-Score: 0.98–0.99
- **Feature Importance:** 
  - `cibil_score` dominates (~80% importance)
  - `loan_term` and `loan_amount` moderate
  - Asset-related features and categorical variables contribute very little
- **Business Insight:** Model decisions are heavily based on CIBIL score, ensuring conservative loan approvals.

### Handling Imbalanced Data (SMOTE)
- Balancing classes improved model learning for minority class.
- Ensures fair prediction for Rejected loans without overfitting.

### Logistic Regression
- Struggles with recall for Approved loans (0.63) even after SMOTE.
- Random Forest clearly outperforms Logistic Regression, especially in predicting approvals.
- Logistic Regression could still be useful for interpretability but not for highest predictive performance.

### Confusion Matrix
- High true positives and true negatives indicate robust classification.
- Very few false approvals, minimizing financial risk.

---

## Conclusion & Key Takeaways
- ✅ **Random Forest is the best performer** with 98.2% accuracy.
- 💡 **CIBIL score** is the dominant feature for loan approval prediction.
- ⚖️ Handling imbalanced data with SMOTE improves learning for minority class.
- 🔍 Model is **conservative**, minimizing risky approvals and supporting data-driven decisions.
- 🚀 **Next Steps:** Hyperparameter tuning, ensemble methods, and feature engineering could further improve performance.

---

**GitHub Repo Suggestion:**
- **Repo Name:** `loan-approval-prediction`
- **Description:** ML pipeline for predicting loan approval using Random Forest, Logistic Regression, and SMOTE-based resampling on an imbalanced dataset.
