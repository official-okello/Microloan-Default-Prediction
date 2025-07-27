# Microloan Default Prediction

A machine learning pipeline to predict the likelihood of loan default using real-world features such as income level, utility payment behavior, and historical borrowing data.

## 📊 Project Overview

This project aims to:
- Predict whether a borrower will default on a loan.
- Use advanced feature engineering to derive behavioral and financial features.
- Apply SMOTE to handle class imbalance.
- Perform dimensionality reduction with PCA.
- Train and optimize an XGBoost model via RandomizedSearchCV.

---

## 🧠 Features Used

- Monthly Income
- Utility Payment Timeliness
- Gender
- Previous Loan Status
- Loan to Income Ratio
- Income Adjusted Loan Ratio
- Payment Discipline Score
- Reliable Borrower Indicator

---

## 🔧 Technologies Used

- Python 3.x
- Pandas & NumPy
- Scikit-learn
- imbalanced-learn
- XGBoost
- Joblib
- Logging

---

## 🛠️ How to Run

1. Clone the repo:
    ```bash
    git clone https://github.com/official-okello/Microloan-Default-Prediction.git
    cd Microloan-Default-Prediction
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the pipeline:
    ```bash
    python src/pipeline.py
    ```

---

## 📁 Project Structure

Microloan Default Prediction/       
├── datasets/       
│ └── simulated_loan_data.csv   
├── saved_models/   
│ └── loan_default_pipeline.joblib  
├── src/    
│ └── pipeline.py   
├── evaluation_report.txt   
├── requirements.txt    
└── README.md   


---

## 📈 Evaluation Metrics

The model is evaluated using:
- Accuracy
- F1 Score
- ROC AUC
- Confusion Matrix
- Classification Report

---

## 📜 License

This project is open-source and available under the MIT License.