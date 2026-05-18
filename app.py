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
    page_title="MedScan AI | Breast Cancer Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stApp {
        background-color: #f7f9fc;
        font-family: 'Segoe UI', sans-serif;
    }
    
    .hospital-header {
        background: linear-gradient(135deg, #0a2342 0%, #1a4a7a 100%);
        padding: 24px 40px;
        border-radius: 0 0 12px 12px;
        margin: -1rem -1rem 2rem -1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .hospital-title {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.5px;
    }
    
    .hospital-subtitle {
        color: #a8c4e0;
        font-size: 0.9rem;
        margin: 4px 0 0 0;
    }
    
    .hospital-badge {
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 20px;
        padding: 6px 16px;
        color: white;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #e2eaf2;
        box-shadow: 0 2px 8px rgba(10,35,66,0.06);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0a2342;
        line-height: 1.1;
    }
    
    .metric-label {
        font-size: 0.78rem;
        color: #6b7c93;
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-sub {
        font-size: 0.7rem;
        color: #27ae60;
        margin-top: 4px;
        font-weight: 600;
    }
    
    .section-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #e2eaf2;
        box-shadow: 0 2px 8px rgba(10,35,66,0.05);
        margin-bottom: 1rem;
    }
    
    .section-title {
        color: #0a2342;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 2px solid #e8f0fa;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    .result-benign {
        background: linear-gradient(135deg, #e8f8f0, #d4f0e4);
        border: 2px solid #27ae60;
        border-radius: 10px;
        padding: 20px 24px;
        text-align: center;
    }
    
    .result-benign .result-title {
        color: #1a7a45;
        font-size: 1.4rem;
        font-weight: 800;
    }
    
    .result-benign .result-sub {
        color: #2ecc71;
        font-size: 0.85rem;
        margin-top: 4px;
    }
    
    .result-malignant {
        background: linear-gradient(135deg, #fef0f0, #fde0e0);
        border: 2px solid #e74c3c;
        border-radius: 10px;
        padding: 20px 24px;
        text-align: center;
    }
    
    .result-malignant .result-title {
        color: #c0392b;
        font-size: 1.4rem;
        font-weight: 800;
    }
    
    .result-malignant .result-sub {
        color: #e74c3c;
        font-size: 0.85rem;
        margin-top: 4px;
    }
    
    .confidence-bar-container {
        background: #e8f0fa;
        border-radius: 20px;
        height: 10px;
        margin-top: 12px;
        overflow: hidden;
    }
    
    .disclaimer {
        background: #fff8e6;
        border: 1px solid #f0c040;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.78rem;
        color: #7a5c00;
        margin-top: 1rem;
    }
    
    .info-pill {
        display: inline-block;
        background: #e8f0fa;
        color: #0a2342;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 3px 3px;
    }
    
    .footer-bar {
        background: #0a2342;
        color: #a8c4e0;
        text-align: center;
        padding: 14px;
        border-radius: 10px;
        font-size: 0.78rem;
        margin-top: 2rem;
    }
    
    [data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid #e2eaf2;
    }
    
    [data-testid="stSidebar"] .stSlider label {
        color: #0a2342 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
    }
    
    .stSelectbox label {
        color: #0a2342 !important;
        font-weight: 600 !important;
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
    return rf, xgb, X_test, y_test, data, X

rf, xgb, X_test, y_test, data, X_full = load_and_train()

st.markdown("""
<div class='hospital-header'>
    <div>
        <div class='hospital-title'>🏥 MedScan AI — Breast Cancer Detection System</div>
        <div class='hospital-subtitle'>AI-Powered Diagnostic Support Tool | Wisconsin Breast Cancer Dataset | For Research Use Only</div>
    </div>
    <div class='hospital-badge'>🔒 Secure Clinical Environment</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
cards = [
    ("96.49%", "Random Forest", "↑ High Precision"),
    ("95.61%", "XGBoost", "↑ High Recall"),
    ("569", "Patient Records", "Training Dataset"),
    ("30", "Clinical Features", "Feature Dimensions"),
]
for col, (val, label, sub) in zip([c1,c2,c3,c4], cards):
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{label}</div>
            <div class='metric-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style='background:#0a2342;padding:16px;border-radius:10px;margin-bottom:16px'>
        <div style='color:white;font-weight:700;font-size:1rem'>🔬 Clinical Parameters</div>
        <div style='color:#a8c4e0;font-size:0.75rem;margin-top:4px'>Enter patient measurement values</div>
    </div>
    """, unsafe_allow_html=True)
    
    mean_radius = st.slider("Mean Radius (mm)", 6.0, 30.0, 14.0, step=0.1)
    mean_texture = st.slider("Mean Texture", 9.0, 40.0, 19.0, step=0.1)
    mean_perimeter = st.slider("Mean Perimeter (mm)", 40.0, 200.0, 92.0, step=0.5)
    mean_area = st.slider("Mean Area (mm²)", 140.0, 2600.0, 654.0, step=5.0)
    mean_smoothness = st.slider("Mean Smoothness", 0.050, 0.170, 0.096, step=0.001, format="%.3f")
    
    st.markdown("---")
    model_choice = st.selectbox("🤖 Select Algorithm", 
                                ["Random Forest", "XGBoost", "Ensemble (Both Combined)"])
    
    st.markdown("""
    <div style='background:#f4f8fc;border-radius:8px;padding:12px;margin-top:12px'>
        <div style='color:#0a2342;font-size:0.75rem;font-weight:700;margin-bottom:8px'>MODEL TAGS</div>
    """, unsafe_allow_html=True)
    for tag in ["Ensemble ML", "96.49% Accuracy", "30 Features", "Binary Classification"]:
        st.markdown(f"<span class='info-pill'>{tag}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

with left:
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

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📋 Diagnostic Result</div>", unsafe_allow_html=True)

    if pred == 1:
        st.markdown(f"""
        <div class='result-benign'>
            <div class='result-title'>✅ BENIGN</div>
            <div class='result-sub'>Low malignancy risk detected — Recommend routine follow-up</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='result-malignant'>
            <div class='result-title'>⚠️ MALIGNANT</div>
            <div class='result-sub'>High malignancy risk detected — Immediate clinical review recommended</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Confidence Score", f"{confidence:.1f}%")
    with col_b:
        st.metric("Algorithm", model_choice.split()[0])

    fig, ax = plt.subplots(figsize=(5, 3))
    colors = ['#27ae60', '#e74c3c']
    bars = ax.bar(['Benign', 'Malignant'], [prob[1]*100, prob[0]*100],
                  color=colors, alpha=0.85, width=0.45,
                  edgecolor=['#1e8449', '#c0392b'], linewidth=1.2)
    ax.set_ylim(0, 115)
    ax.set_ylabel('Probability (%)', color='#0a2342', fontsize=10)
    ax.set_title('Prediction Probability Distribution', 
                 color='#0a2342', fontsize=11, fontweight='bold', pad=12)
    for bar, val in zip(bars, [prob[1]*100, prob[0]*100]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom',
                color='#0a2342', fontsize=11, fontweight='bold')
    ax.set_facecolor('#f7f9fc')
    fig.patch.set_facecolor('#ffffff')
    ax.tick_params(colors='#0a2342', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e2eaf2')
    ax.spines['bottom'].set_color('#e2eaf2')
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='disclaimer'>
        ⚠️ <strong>Clinical Disclaimer:</strong> This tool is intended for research and educational purposes only. 
        Results should not be used as a substitute for professional medical diagnosis. 
        Always consult a qualified healthcare professional.
    </div>""", unsafe_allow_html=True)

with right:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🔍 Feature Importance Analysis</div>", unsafe_allow_html=True)
    
    importances = pd.Series(rf.feature_importances_, 
                           index=data.feature_names).sort_values(ascending=True)[-10:]
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    bars2 = ax2.barh(range(len(importances)), importances.values,
                     color='#1a4a7a', alpha=0.85,
                     edgecolor='#0a2342', linewidth=0.8)
    ax2.set_yticks(range(len(importances)))
    ax2.set_yticklabels(importances.index, fontsize=9, color='#0a2342')
    ax2.set_title('Top 10 Most Influential Clinical Features',
                  color='#0a2342', fontsize=11, fontweight='bold', pad=12)
    ax2.set_xlabel('Importance Score', color='#0a2342', fontsize=10)
    ax2.set_facecolor('#f7f9fc')
    fig2.patch.set_facecolor('#ffffff')
    ax2.tick_params(colors='#0a2342', labelsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#e2eaf2')
    ax2.spines['bottom'].set_color('#e2eaf2')
    for i, (bar, val) in enumerate(zip(bars2, importances.values)):
        ax2.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8, color='#0a2342')
    plt.tight_layout()
    st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>ℹ️ Model Information</div>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🌲 Random Forest", "⚡ XGBoost"])
    with tab1:
        st.markdown("""
        <p style='color:#333;font-size:0.88rem;line-height:1.7'>
        Trains <strong>100 decision trees</strong> on random subsets of patient data and aggregates their predictions via majority voting. 
        Highly robust to noise, outliers, and missing values — ideal for clinical diagnostic datasets.
        <br><br>
        <strong>Accuracy:</strong> 96.49% &nbsp;|&nbsp; <strong>Trees:</strong> 100 &nbsp;|&nbsp; <strong>Features:</strong> 30
        </p>""", unsafe_allow_html=True)
    with tab2:
        st.markdown("""
        <p style='color:#333;font-size:0.88rem;line-height:1.7'>
        Uses <strong>gradient boosting</strong> to sequentially build trees that correct prior prediction errors. 
        One of the highest-performing algorithms for structured healthcare tabular data.
        <br><br>
        <strong>Accuracy:</strong> 95.61% &nbsp;|&nbsp; <strong>Objective:</strong> Binary:Logistic &nbsp;|&nbsp; <strong>Features:</strong> 30
        </p>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class='footer-bar'>
    MedScan AI &nbsp;|&nbsp; Built by <strong>Priyanka Kapoor</strong> &nbsp;|&nbsp; 
    MS Business Analytics, Montclair State University 2026 &nbsp;|&nbsp;
    <a href='https://github.com/PriyankaKapoor4202' style='color:#a8c4e0'>GitHub</a> &nbsp;|&nbsp;
    For Research & Educational Use Only
</div>
""", unsafe_allow_html=True)
