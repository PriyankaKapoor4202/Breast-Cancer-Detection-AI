import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

st.title("Breast Cancer Detection AI")
st.write("Predicts malignant vs benign using Random Forest and XGBoost")

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

xgb = XGBClassifier(random_state=42, eval_metric='logloss')
xgb.fit(X_train, y_train)

st.subheader("Model Performance")
st.write(f"Random Forest Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.2%}")
st.write(f"XGBoost Accuracy: {accuracy_score(y_test, xgb.predict(X_test)):.2%}")

st.subheader("Try a Prediction")
mean_radius = st.slider("Mean Radius", 6.0, 30.0, 14.0)
mean_texture = st.slider("Mean Texture", 9.0, 40.0, 19.0)
mean_perimeter = st.slider("Mean Perimeter", 40.0, 200.0, 92.0)

input_data = pd.DataFrame([X.mean().values], columns=data.feature_names)
input_data['mean radius'] = mean_radius
input_data['mean texture'] = mean_texture
input_data['mean perimeter'] = mean_perimeter

prediction = rf.predict(input_data)[0]
st.subheader("Prediction")
st.write("Benign" if prediction == 1 else "Malignant")
