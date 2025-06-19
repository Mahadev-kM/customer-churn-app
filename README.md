# 📉 Customer Churn Prediction App

A machine learning-powered web app to predict whether a customer is likely to churn based on their service usage and account information. Built using Python, scikit-learn, and Streamlit.

---

## 🚀 Project Overview

Customer churn is a major concern for subscription-based businesses like telecom providers. This project focuses on predicting customer churn so that companies can proactively engage at-risk customers and reduce revenue loss.

✅ **Goal:** Build and deploy a predictive model that classifies customers as **churned** or **retained** based on historical data.

---

## 🧠 Machine Learning Workflow

### 1. 📁 Dataset
- **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Size:** ~7,000 rows and 21 columns
- **Target Variable:** `Churn` (Yes/No)

### 2. 🔍 Exploratory Data Analysis (EDA)
- Visualized churn distribution
- Analyzed key features like contract type, tenure, and payment method
- Identified high-churn segments (e.g., monthly contracts, fiber internet)

### 3. 🛠️ Data Preprocessing
- Handled missing values
- Label encoded binary categorical columns
- One-hot encoded multiclass categorical columns
- Normalized numerical columns
- Addressed class imbalance using **SMOTE**

### 4. 🤖 Model Training
Trained and evaluated multiple models:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost ✅ (Best performing)

### 5. 📊 Model Evaluation
- Accuracy: **82%**
- Precision: **79%**
- Recall: **78%**
- F1-Score: **78%**
- ROC AUC: **85%**

### 6. 🌐 Deployment
- Built an interactive UI using **Streamlit**
- Saved model and pipeline using **Pickle**
- Allows real-time predictions by entering customer details manually

---

## 🧪 How to Run the App Locally

### 🔧 Prerequisites
- Python 3.8+
- pip installed packages (see `requirements.txt`)

### 🖥️ Installation & Execution

```bash
# Clone the repository
git clone https://github.com/Mahadev-kM/customer-churn-app.git
cd customer-churn-app

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
