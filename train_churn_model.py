# train_churn_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Step 1: Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# Step 2: Data Cleaning
df.drop(['customerID'], axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Step 3: Encode target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Step 4: Encode 'Contract' as one-hot (same as Streamlit app)
df = pd.get_dummies(df, columns=['Contract'], drop_first=True)

# Step 5: Feature selection
selected_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract_One year', 'Contract_Two year']
X = df[selected_features]
y = df['Churn']

# Step 6: Scale numeric features
scaler = StandardScaler()
X[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(X[['tenure', 'MonthlyCharges', 'TotalCharges']])

# Step 7: Train-test split (optional for training only)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 9: Save model to .pkl
with open("churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as churn_model.pkl")
