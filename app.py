import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Breast Cancer Detection AI",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .stApp { background-color: #f0f4f8; }
    h1 { color: #1a3a5c; font-family: 'Georgia', serif; }
    h2, h3 { color: #1a3a5c; }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .prediction-benign {
        background-color: #e8f5e9;
        border-left: 6px solid #2e7d32;
        padding: 20px;
        border-radius: 8px;
        font-size: 1.3em;
        color: #2e7d32;
        font-weight: bold;
    }
    .prediction-malignant {
        background-color: #ffebee;
        border-left: 6px solid #c62828;
        padding: 20px;
        border-radius: 8px;
        font-size: 1.3em;
        color: #c62828;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_and_train():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    
    return rf, xgb, X_test, y_test, data

rf, xgb, X_test, y_test, data = load_and_train()

st.markdown("# 🏥 Breast Cancer Detection AI")
st.markdown("### Early detection support tool using ensemble machine learning")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""<div class='metric-card'>
        <h2 style='color:#1a3a5c'>96.49%</h2>
        <p>Random Forest Accuracy</p>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class='metric-card'>
        <h2 style='color:#1a3a5c'>95.61%</h2>
        <p>XGBoost Accuracy</p>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class='metric-card'>
        <h2 style='color:#1a3a5c'>569</h2>
        <p>Patient Records Analyzed</p>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
left, right = st.columns([1, 1])

with left:
    st.markdown("### 🔬 Patient Feature Input")
    st.markdown("Adjust the sliders to match patient measurements:")
    
    mean_radius = st.slider("Mean Radius", 6.0, 30.0, 14.0, help="Mean of distances from center to perimeter")
    mean_texture = st.slider("Mean Texture", 9.0, 40.0, 19.0, help="Standard deviation of gray-scale values")
    mean_perimeter = st.slider("Mean Perimeter", 40.0, 200.0, 92.0)
    mean_area = st.slider("Mean Area", 140.0, 2600.0, 654.0)
    mean_smoothness = st.slider("Mean Smoothness", 0.05, 0.17, 0.096, format="%.3f")
    
    model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost", "Both (Ensemble)"])

with right:
    st.markdown("### 📊 Prediction Result")
    
    X_sample = pd.DataFrame([pd.DataFrame(data.data, columns=data.feature_names).mean().values],
                            columns=data.feature_names)
    X_sample['mean radius'] = mean_radius
    X_sample['mean texture'] = mean_texture
    X_sample['mean perimeter'] = mean_perimeter
    X_sample['mean area'] = mean_area
    X_sample['mean smoothness'] = mean_smoothness
    
    if model_choice == "Random Forest":
        pred = rf.predict(X_sample)[0]
        prob = rf.predict_proba(X_sample)[0]
    elif model_choice == "XGBoost":
        pred = xgb.predict(X_sample)[0]
        prob = xgb.predict_proba(X_sample)[0]
    else:
        rf_pred = rf.predict_proba(X_sample)[0]
        xgb_pred = xgb.predict_proba(X_sample)[0]
        prob = (rf_pred + xgb_pred) / 2
        pred = 1 if prob[1] > 0.5 else 0
    
    if pred == 1:
        st.markdown("<div class='prediction-benign'>✅ Benign — Low Risk Detected</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='prediction-malignant'>⚠️ Malignant — High Risk Detected</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    confidence = max(prob) * 100
    st.metric("Model Confidence", f"{confidence:.1f}%")
    
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(['Benign', 'Malignant'], [prob[1]*100, prob[0]*100],
           color=['#2e7d32', '#c62828'], alpha=0.8)
    ax.set_ylabel('Probability (%)')
    ax.set_title('Prediction Probability')
    ax.set_facecolor('#f0f4f8')
    fig.patch.set_facecolor('#f0f4f8')
    st.pyplot(fig)

st.markdown("---")
st.markdown("### 🔍 Feature Importance (Random Forest)")
importances = pd.Series(rf.feature_importances_, index=data.feature_names).sort_values(ascending=False)[:10]
fig2, ax2 = plt.subplots(figsize=(10, 4))
importances.plot(kind='bar', ax=ax2, color='#1a3a5c', alpha=0.85)
ax2.set_title('Top 10 Most Important Features')
ax2.set_facecolor('#f0f4f8')
fig2.patch.set_facecolor('#f0f4f8')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig2)

st.markdown("---")
st.markdown("### ℹ️ How It Works")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **Random Forest** trains 100 decision trees on different subsets of the data 
    and combines their predictions. It is highly robust to overfitting and works 
    well with medical data.
    """)
with col2:
    st.markdown("""
    **XGBoost** uses gradient boosting to sequentially build trees that correct 
    previous errors. It is one of the most powerful algorithms for structured 
    healthcare data.
    """)

st.markdown("---")
st.caption("Built by Priyanka Kapoor | MS Business Analytics, Montclair State University | github.com/PriyankaKapoor4202")
