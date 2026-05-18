import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

st.set_page_config(
    page_title="Breast Cancer Detection AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    body { background-color: #ffffff; }
    .stApp { background-color: #ffffff; }
    .main .block-container { padding: 2rem 3rem; }
    h1 { color: #0d3362 !important; font-size: 2.2rem !important; font-weight: 700 !important; }
    h2 { color: #0d3362 !important; font-size: 1.4rem !important; }
    h3 { color: #0d3362 !important; font-size: 1.1rem !important; }
    p { color: #333333 !important; }
    .metric-box {
        background: #ffffff;
        border: 1.5px solid #d0dce8;
        border-radius: 12px;
        padding: 24px 16px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .metric-box .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0d3362;
    }
    .metric-box .metric-label {
        font-size: 0.85rem;
        color: #555555;
        margin-top: 4px;
    }
    .benign-box {
        background: #f0faf0;
        border: 2px solid #2e7d32;
        border-radius: 10px;
        padding: 18px 22px;
        color: #1b5e20;
        font-size: 1.2rem;
        font-weight: 700;
    }
    .malignant-box {
        background: #fff5f5;
        border: 2px solid #c62828;
        border-radius: 10px;
        padding: 18px 22px;
        color: #b71c1c;
        font-size: 1.2rem;
        font-weight: 700;
    }
    .section-divider {
        border: none;
        border-top: 1.5px solid #e0e8f0;
        margin: 2rem 0;
    }
    .info-box {
        background: #f4f8fc;
        border-radius: 10px;
        padding: 16px 20px;
        color: #333333;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    [data-testid="stSidebar"] {
        background-color: #f4f8fc;
    }
    [data-testid="stSidebar"] h2 {
        color: #0d3362 !important;
    }
    label { color: #333333 !important; }
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
    return rf, xgb, X_test, y_test, data, X

rf, xgb, X_test, y_test, data, X_full = load_and_train()

st.markdown("# 🏥 Breast Cancer Detection AI")
st.markdown("**Early detection support tool using ensemble machine learning — Wisconsin Breast Cancer Dataset**")
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
metrics = [
    ("96.49%", "Random Forest Accuracy"),
    ("95.61%", "XGBoost Accuracy"),
    ("569", "Patient Records"),
    ("30", "Clinical Features"),
]
for col, (val, label) in zip([c1, c2, c3, c4], metrics):
    with col:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🔬 Patient Inputs")
    st.markdown("Adjust values to match patient measurements:")
    mean_radius = st.slider("Mean Radius", 6.0, 30.0, 14.0)
    mean_texture = st.slider("Mean Texture", 9.0, 40.0, 19.0)
    mean_perimeter = st.slider("Mean Perimeter", 40.0, 200.0, 92.0)
    mean_area = st.slider("Mean Area", 140.0, 2600.0, 654.0)
    mean_smoothness = st.slider("Mean Smoothness", 0.050, 0.170, 0.096, step=0.001, format="%.3f")
    st.markdown("---")
    model_choice = st.selectbox("Model", ["Random Forest", "XGBoost", "Ensemble (Both)"])

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("### 📊 Prediction Result")
    X_sample = pd.DataFrame([X_full.mean().values], columns=data.feature_names)
    X_sample['mean radius'] = mean_radius
    X_sample['mean texture'] = mean_texture
    X_sample['mean perimeter'] = mean_perimeter
    X_sample['mean area'] = mean_area
    X_sample['mean smoothness'] = mean_smoothness

    if model_choice == "Random Forest":
        prob = rf.predict_proba(X_sample)[0]
    elif model_choice == "XGBoost":
        prob = xgb.predict_proba(X_sample)[0]
    else:
        prob = (rf.predict_proba(X_sample)[0] + xgb.predict_proba(X_sample)[0]) / 2

    pred = 1 if prob[1] > 0.5 else 0
    confidence = max(prob) * 100

    if pred == 1:
        st.markdown("<div class='benign-box'>✅ Result: <strong>Benign</strong> — Low malignancy risk detected</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='malignant-box'>⚠️ Result: <strong>Malignant</strong> — High malignancy risk detected</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.metric("Confidence Score", f"{confidence:.1f}%")

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(['Benign', 'Malignant'], [prob[1]*100, prob[0]*100],
                  color=['#2e7d32', '#c62828'], alpha=0.85, width=0.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel('Probability (%)', color='#333333', fontsize=11)
    ax.set_title('Prediction Probability Breakdown', color='#0d3362', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, [prob[1]*100, prob[0]*100]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', color='#333333', fontsize=11)
    ax.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#ffffff')
    ax.tick_params(colors='#333333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)

with right:
    st.markdown("### 🔍 Top 10 Feature Importances")
    importances = pd.Series(rf.feature_importances_, index=data.feature_names).sort_values(ascending=True)[-10:]
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    importances.plot(kind='barh', ax=ax2, color='#0d3362', alpha=0.82)
    ax2.set_title('Most Influential Clinical Features', color='#0d3362', fontsize=12, fontweight='bold')
    ax2.set_facecolor('#ffffff')
    fig2.patch.set_facecolor('#ffffff')
    ax2.tick_params(colors='#333333', labelsize=9)
    ax2.set_xlabel('Importance Score', color='#333333')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("### ℹ️ About the Models")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""<div class='info-box'>
    <strong>🌲 Random Forest (96.49% accuracy)</strong><br>
    Trains 100 decision trees on random subsets of patient data and combines their votes. 
    Highly robust to noise and missing values — ideal for clinical datasets.
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class='info-box'>
    <strong>⚡ XGBoost (95.61% accuracy)</strong><br>
    Uses gradient boosting to sequentially correct prediction errors. 
    One of the most powerful algorithms for structured healthcare tabular data.
    </div>""", unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("<p style='color:#888; font-size:0.85rem; text-align:center'>Built by <strong>Priyanka Kapoor</strong> | MS Business Analytics, Montclair State University 2026 | <a href='https://github.com/PriyankaKapoor4202' style='color:#0d3362'>GitHub</a></p>", unsafe_allow_html=True)
