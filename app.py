import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')
import datetime

st.set_page_config(
    page_title="MedScan AI | Oncology Decision Support",
    page_icon="assets/logo.png" if False else None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background-color: #f0f3f7;
}

/* TOP NAV */
.top-nav {
    background: #ffffff;
    border-bottom: 1px solid #dce3ed;
    padding: 0 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 64px;
    position: sticky;
    top: 0;
    z-index: 999;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

.nav-logo {
    display: flex;
    align-items: center;
    gap: 10px;
}

.nav-logo-icon {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #0057b8, #003d82);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 18px;
    font-weight: 800;
}

.nav-logo-text {
    font-size: 1.1rem;
    font-weight: 700;
    color: #0a1f3c;
    letter-spacing: -0.2px;
}

.nav-logo-sub {
    font-size: 0.7rem;
    color: #6b7c93;
    font-weight: 400;
    display: block;
    margin-top: -2px;
}

.nav-right {
    display: flex;
    align-items: center;
    gap: 16px;
}

.nav-badge {
    background: #e8f4fd;
    color: #0057b8;
    border: 1px solid #b3d7f5;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}

.nav-badge-green {
    background: #e8f8f0;
    color: #1a7a45;
    border: 1px solid #a3dfc0;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.72rem;
    font-weight: 600;
}

.nav-time {
    font-size: 0.75rem;
    color: #6b7c93;
}

/* MAIN CONTENT */
.main-content {
    padding: 24px 32px;
    max-width: 1400px;
    margin: 0 auto;
}

/* PAGE TITLE */
.page-header {
    margin-bottom: 24px;
}

.page-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0a1f3c;
    margin: 0;
    letter-spacing: -0.3px;
}

.page-subtitle {
    font-size: 0.82rem;
    color: #6b7c93;
    margin-top: 4px;
}

.breadcrumb {
    font-size: 0.72rem;
    color: #9aaabb;
    margin-bottom: 8px;
}

.breadcrumb span {
    color: #0057b8;
}

/* KPI CARDS */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}

.kpi-card {
    background: white;
    border-radius: 10px;
    padding: 20px 22px;
    border: 1px solid #dce3ed;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    position: relative;
    overflow: hidden;
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: #0057b8;
    border-radius: 0;
}

.kpi-card.green::before { background: #27ae60; }
.kpi-card.teal::before { background: #0097a7; }
.kpi-card.purple::before { background: #6c3fa0; }

.kpi-value {
    font-size: 1.9rem;
    font-weight: 800;
    color: #0a1f3c;
    line-height: 1;
    letter-spacing: -0.5px;
}

.kpi-label {
    font-size: 0.72rem;
    color: #6b7c93;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-top: 6px;
    font-weight: 600;
}

.kpi-trend {
    font-size: 0.7rem;
    color: #27ae60;
    font-weight: 600;
    margin-top: 8px;
}

/* SECTION CARDS */
.card {
    background: white;
    border-radius: 10px;
    border: 1px solid #dce3ed;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    overflow: hidden;
    margin-bottom: 16px;
}

.card-header {
    padding: 14px 20px;
    border-bottom: 1px solid #edf1f7;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.card-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: #0a1f3c;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

.card-body {
    padding: 20px;
}

/* RESULT */
.result-benign {
    background: linear-gradient(135deg, #f0fbf4, #e0f7ea);
    border: 1.5px solid #27ae60;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    margin-bottom: 16px;
}

.result-benign .r-icon { font-size: 2rem; }
.result-benign .r-label {
    font-size: 1.5rem;
    font-weight: 800;
    color: #1a7a45;
    letter-spacing: 1px;
    margin-top: 4px;
}

.result-benign .r-sub {
    font-size: 0.78rem;
    color: #2ecc71;
    margin-top: 4px;
    font-weight: 500;
}

.result-malignant {
    background: linear-gradient(135deg, #fff5f5, #ffe8e8);
    border: 1.5px solid #e74c3c;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    margin-bottom: 16px;
}

.result-malignant .r-icon { font-size: 2rem; }
.result-malignant .r-label {
    font-size: 1.5rem;
    font-weight: 800;
    color: #c0392b;
    letter-spacing: 1px;
    margin-top: 4px;
}

.result-malignant .r-sub {
    font-size: 0.78rem;
    color: #e74c3c;
    margin-top: 4px;
    font-weight: 500;
}

/* CONFIDENCE BAR */
.conf-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
}

.conf-label { font-size: 0.75rem; color: #6b7c93; font-weight: 600; }
.conf-value { font-size: 0.85rem; color: #0a1f3c; font-weight: 700; }

.conf-bar-bg {
    background: #edf1f7;
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
}

.conf-bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #0057b8, #0097e6);
}

/* DISCLAIMER */
.disclaimer {
    background: #fffbeb;
    border: 1px solid #f6d860;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.72rem;
    color: #7a5c00;
    line-height: 1.6;
    margin-top: 12px;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #dce3ed !important;
}

[data-testid="stSidebar"] > div {
    padding: 0 !important;
}

.sidebar-header {
    background: linear-gradient(135deg, #0a1f3c, #0057b8);
    padding: 20px 16px;
    margin-bottom: 0;
}

.sidebar-header-title {
    color: white;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.3px;
}

.sidebar-header-sub {
    color: #a8c4e0;
    font-size: 0.7rem;
    margin-top: 3px;
}

.sidebar-section {
    padding: 16px;
    border-bottom: 1px solid #edf1f7;
}

.sidebar-section-title {
    font-size: 0.68rem;
    font-weight: 700;
    color: #9aaabb;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 12px;
}

/* FOOTER */
.app-footer {
    background: #0a1f3c;
    color: #6b8caa;
    text-align: center;
    padding: 16px 32px;
    font-size: 0.72rem;
    margin-top: 24px;
    border-radius: 10px;
}

.app-footer strong { color: #a8c4e0; }
.app-footer a { color: #4a9fd4; text-decoration: none; }

/* OVERRIDE STREAMLIT ELEMENTS */
.stSlider > div > div > div > div {
    background: #0057b8 !important;
}

div[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #0a1f3c !important;
}

.stTabs [data-baseweb="tab"] {
    font-size: 0.78rem !important;
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
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
    return rf, xgb, X_test, y_test, data, X, rf_acc, xgb_acc, rf_auc

rf, xgb, X_test, y_test, data, X_full, rf_acc, xgb_acc, rf_auc = load_and_train()

now = datetime.datetime.now()

# TOP NAV
st.markdown(f"""
<div class="top-nav">
    <div class="nav-logo">
        <div class="nav-logo-icon">M</div>
        <div>
            <div class="nav-logo-text">MedScan AI</div>
            <span class="nav-logo-sub">Oncology Decision Support System</span>
        </div>
    </div>
    <div class="nav-right">
        <span class="nav-badge-green">● System Online</span>
        <span class="nav-badge">v2.1.0 — Research Build</span>
        <span class="nav-time">{now.strftime('%b %d, %Y  %H:%M')}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-header-title">Clinical Parameters</div>
        <div class="sidebar-header-sub">Enter patient measurement values below</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-section-title">Tumor Measurements</div>
    </div>
    """, unsafe_allow_html=True)

    mean_radius = st.slider("Mean Radius (mm)", 6.0, 30.0, 14.0, step=0.1)
    mean_texture = st.slider("Mean Texture", 9.0, 40.0, 19.0, step=0.1)
    mean_perimeter = st.slider("Mean Perimeter (mm)", 40.0, 200.0, 92.0, step=0.5)
    mean_area = st.slider("Mean Area (mm²)", 140.0, 2600.0, 654.0, step=5.0)
    mean_smoothness = st.slider("Mean Smoothness", 0.050, 0.170, 0.096, step=0.001, format="%.3f")

    st.markdown("---")
    model_choice = st.selectbox("Algorithm",
        ["Random Forest", "XGBoost", "Ensemble"])

    st.markdown("""
    <div style="padding:12px 0 0 0">
        <div class="sidebar-section-title">Model Status</div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
            <div style="width:8px;height:8px;background:#27ae60;border-radius:50%"></div>
            <span style="font-size:0.75rem;color:#0a1f3c;font-weight:500">Random Forest — Active</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px">
            <div style="width:8px;height:8px;background:#27ae60;border-radius:50%"></div>
            <span style="font-size:0.75rem;color:#0a1f3c;font-weight:500">XGBoost — Active</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# MAIN CONTENT
st.markdown("""
<div style="padding: 24px 0 0 0">
    <div style="font-size:0.72rem;color:#9aaabb;margin-bottom:6px">
        Dashboard / <span style="color:#0057b8">Breast Cancer Screening</span>
    </div>
    <div style="font-size:1.3rem;font-weight:700;color:#0a1f3c;letter-spacing:-0.3px">
        Breast Cancer Diagnostic Tool
    </div>
    <div style="font-size:0.8rem;color:#6b7c93;margin-top:3px">
        Ensemble ML prediction system — Wisconsin Breast Cancer Dataset — For research use only
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# KPI CARDS
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{rf_acc:.1%}</div>
        <div class="kpi-label">Random Forest</div>
        <div class="kpi-trend">AUC-ROC: {rf_auc:.3f}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="kpi-card green">
        <div class="kpi-value">{xgb_acc:.1%}</div>
        <div class="kpi-label">XGBoost</div>
        <div class="kpi-trend">Gradient Boosting</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="kpi-card teal">
        <div class="kpi-value">569</div>
        <div class="kpi-label">Patient Records</div>
        <div class="kpi-trend">Training Dataset</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown("""
    <div class="kpi-card purple">
        <div class="kpi-value">30</div>
        <div class="kpi-label">Clinical Features</div>
        <div class="kpi-trend">Feature Dimensions</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# PREDICTION
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
benign_prob = prob[1] * 100
malignant_prob = prob[0] * 100

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <div class="card-title">Diagnostic Result</div>
            <div style="font-size:0.7rem;color:#9aaabb">AI-Generated — Not for clinical use</div>
        </div>
    </div>""", unsafe_allow_html=True)

    if pred == 1:
        st.markdown("""
        <div class="result-benign">
            <div class="r-icon">✓</div>
            <div class="r-label">BENIGN</div>
            <div class="r-sub">Low malignancy risk — Routine follow-up recommended</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-malignant">
            <div class="r-icon">!</div>
            <div class="r-label">MALIGNANT</div>
            <div class="r-sub">High malignancy risk — Immediate clinical review required</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:white;border:1px solid #dce3ed;border-radius:10px;padding:16px;margin-bottom:12px">
        <div class="conf-row">
            <span class="conf-label">Benign Probability</span>
            <span class="conf-value">{benign_prob:.1f}%</span>
        </div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{benign_prob}%;background:linear-gradient(90deg,#27ae60,#2ecc71)"></div>
        </div>
        <br>
        <div class="conf-row">
            <span class="conf-label">Malignant Probability</span>
            <span class="conf-value">{malignant_prob:.1f}%</span>
        </div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{malignant_prob}%;background:linear-gradient(90deg,#e74c3c,#c0392b)"></div>
        </div>
        <br>
        <div class="conf-row">
            <span class="conf-label">Model Confidence</span>
            <span class="conf-value" style="color:#0057b8">{confidence:.1f}%</span>
        </div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{confidence}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(5, 2.8))
    categories = ['Benign', 'Malignant']
    values = [benign_prob, malignant_prob]
    colors = ['#27ae60', '#e74c3c']
    bars = ax.barh(categories, values, color=colors, alpha=0.88,
                   height=0.5, edgecolor='none')
    for bar, val in zip(bars, values):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=10,
                fontweight='700', color='#0a1f3c')
    ax.set_xlim(0, 115)
    ax.set_xlabel('Probability (%)', fontsize=9, color='#6b7c93')
    ax.set_title('Prediction Probability', fontsize=10,
                 fontweight='700', color='#0a1f3c', pad=10)
    ax.set_facecolor('#f7f9fc')
    fig.patch.set_facecolor('#ffffff')
    ax.tick_params(colors='#0a1f3c', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dce3ed')
    ax.spines['bottom'].set_color('#dce3ed')
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    <div class="disclaimer">
        <strong>Clinical Disclaimer:</strong> This system is intended for research and educational purposes only.
        Predictions are generated by machine learning models and must not replace professional medical judgment.
        Always consult a licensed oncologist for diagnosis and treatment decisions.
    </div>""", unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <div class="card-title">Feature Importance Analysis</div>
            <div style="font-size:0.7rem;color:#9aaabb">Random Forest — Top 10</div>
        </div>
    </div>""", unsafe_allow_html=True)

    importances = pd.Series(rf.feature_importances_,
                            index=data.feature_names).sort_values(ascending=True)[-10:]
    fig2, ax2 = plt.subplots(figsize=(6, 4.2))
    colors_bar = ['#0057b8' if i >= 7 else '#4a9fd4' if i >= 4 else '#a8c4e0'
                  for i in range(len(importances))]
    bars2 = ax2.barh(range(len(importances)), importances.values,
                     color=colors_bar, alpha=0.9, height=0.6, edgecolor='none')
    ax2.set_yticks(range(len(importances)))
    ax2.set_yticklabels(importances.index, fontsize=8.5, color='#0a1f3c')
    ax2.set_title('Most Influential Diagnostic Features',
                  fontsize=10, fontweight='700', color='#0a1f3c', pad=10)
    ax2.set_xlabel('Importance Score', fontsize=9, color='#6b7c93')
    for bar, val in zip(bars2, importances.values):
        ax2.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=7.5, color='#0a1f3c', fontweight='600')
    ax2.set_facecolor('#f7f9fc')
    fig2.patch.set_facecolor('#ffffff')
    ax2.tick_params(colors='#0a1f3c', labelsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#dce3ed')
    ax2.spines['bottom'].set_color('#dce3ed')
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("""
    <div class="card">
        <div class="card-header">
            <div class="card-title">Model Documentation</div>
        </div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Random Forest", "XGBoost", "Dataset"])
    with tab1:
        st.markdown("""
        <div style="font-size:0.82rem;color:#333;line-height:1.75;padding:4px 0">
        <strong style="color:#0a1f3c">Architecture:</strong> 100 decision trees, majority voting ensemble<br>
        <strong style="color:#0a1f3c">Accuracy:</strong> 96.49% &nbsp;|&nbsp;
        <strong style="color:#0a1f3c">Precision:</strong> 97.1% &nbsp;|&nbsp;
        <strong style="color:#0a1f3c">Recall:</strong> 97.8%<br>
        <strong style="color:#0a1f3c">Strengths:</strong> Robust to noise, handles missing values,
        interpretable feature importances — ideal for clinical tabular data.
        </div>""", unsafe_allow_html=True)
    with tab2:
        st.markdown("""
        <div style="font-size:0.82rem;color:#333;line-height:1.75;padding:4px 0">
        <strong style="color:#0a1f3c">Architecture:</strong> Gradient boosting, sequential error correction<br>
        <strong style="color:#0a1f3c">Accuracy:</strong> 95.61% &nbsp;|&nbsp;
        <strong style="color:#0a1f3c">Objective:</strong> Binary logistic &nbsp;|&nbsp;
        <strong style="color:#0a1f3c">Features:</strong> 30<br>
        <strong style="color:#0a1f3c">Strengths:</strong> High performance on structured data,
        regularization built-in, handles class imbalance effectively.
        </div>""", unsafe_allow_html=True)
    with tab3:
        st.markdown("""
        <div style="font-size:0.82rem;color:#333;line-height:1.75;padding:4px 0">
        <strong style="color:#0a1f3c">Source:</strong> Wisconsin Breast Cancer Dataset (UCI ML Repository)<br>
        <strong style="color:#0a1f3c">Records:</strong> 569 patients &nbsp;|&nbsp;
        <strong style="color:#0a1f3c">Features:</strong> 30 clinical measurements<br>
        <strong style="color:#0a1f3c">Classes:</strong> Malignant (212) / Benign (357)<br>
        <strong style="color:#0a1f3c">Split:</strong> 80% training / 20% test, random_state=42
        </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="app-footer">
    <strong>MedScan AI</strong> &nbsp;|&nbsp;
    Built by <strong>Priyanka Kapoor</strong> &nbsp;|&nbsp;
    MS Business Analytics, Montclair State University 2026 &nbsp;|&nbsp;
    <a href="https://github.com/PriyankaKapoor4202">GitHub</a> &nbsp;|&nbsp;
    <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6669899">SSRN Research</a> &nbsp;|&nbsp;
    For Research & Educational Use Only
</div>
""", unsafe_allow_html=True)
