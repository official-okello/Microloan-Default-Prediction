
# 🧩 Microloan Default Prediction

## 🎯 Project Title
**AI-Powered Microloan Default Prediction for Kenya's Informal Sector**

---

## 🎯 Problem Statement

In Kenya, a significant portion of the population operates within the informal sector—self-employed workers, *jua kali* artisans, *boda-boda* riders, and small-scale traders. Despite contributing substantially to the economy, these individuals often lack access to affordable credit due to the absence of reliable financial histories.

Microfinance institutions (MFIs) and digital lenders attempt to fill this gap but face high default rates due to rudimentary or non-existent credit scoring systems. This results in high interest rates, loan caps, and exclusion of high-risk (but potentially creditworthy) individuals.

**This project aims to build a machine learning-based credit risk assessment model** that predicts the likelihood of a borrower defaulting on a loan. By using alternative data sources such as mobile money transactions, utility payments, and behavioral indicators, the model can support smarter, data-driven lending decisions that are more inclusive and sustainable.

---

## 📦 Data Design

### 📁 1. Target Variable
- `default`: Binary classification label  
  - `1` → borrower defaulted on the loan  
  - `0` → borrower repaid successfully  

---

### 📊 2. Input Features

| Feature | Type | Description |
|--------|------|-------------|
| `age` | Numeric | Borrower's age |
| `gender` | Categorical | Male or Female |
| `region` | Categorical | Location (e.g., Nairobi, Kisumu) |
| `monthly_income` | Numeric | Estimated income from primary activity |
| `loan_amount` | Numeric | Amount borrowed |
| `repayment_period` | Categorical | Loan term in days (7, 14, 30, 60) |
| `has_previous_loan` | Binary | Has previously borrowed (1) or not (0) |
| `transaction_volume_mtd` | Numeric | Mobile money transactions (count per month) |
| `avg_mpesa_balance` | Numeric | Average M-PESA balance over last 30 days |
| `utility_payment_timeliness` | Ordinal | "early", "on-time", or "late" bill payments |

---

### 🏷️ 3. Feature Engineering Ideas
- `loan_to_income_ratio = loan_amount / monthly_income`
- `payment_discipline_score` → Map "early" to 1, "on-time" to 0.5, "late" to 0
- `income_stability_index` → Rolling average or standard deviation if time-series data is used

---

### 🛠️ 4. Data Sources
Since real-world financial data is difficult to access, the initial dataset will be:
- **Simulated based on realistic assumptions** using economic data, M-PESA usage trends, and patterns observed in public datasets (e.g., LendingClub, Kiva)
- Augmented in future stages with real datasets from:
  - [FSD Kenya](https://fsdkenya.org/)
  - [CGAP](https://www.cgap.org/)
  - World Bank FinAccess Reports
  - Partner SACCOs or fintechs

---

### ⚠️ 5. Challenges in Data
- Class imbalance: Defaults are rarer than repayments
- Data sparsity in informal sectors
- Bias: Region or gender-based lending discrimination must be carefully monitored
- Privacy and ethics: Use only anonymized or simulated data in public prototypes
