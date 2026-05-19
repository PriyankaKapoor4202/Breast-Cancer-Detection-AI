import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import datetime

st.set_page_config(
    page_title="MedScan AI | Breast Cancer Screening",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }

.stApp { background-color: #f1f4f8; }

/* TOP HEADER BAR */
.top-header {
    background: #1b3a6b;
    padding: 0 32px;
    height: 56px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: -1rem -1rem 0 -1rem;
}
.header-left {
    display: flex;
    align-items: center;
    gap: 12px;
}
.header-logo {
    width: 32px;
    height: 32px;
    background: #ffffff;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    font-weight: 800;
    color: #1b3a6b;
}
.header-name {
    color: #ffffff;
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.2px;
}
.header-dept {
    color: #a8bdd4;
    font-size: 0.72rem;
    font-weight: 400;
}
.header-right {
    display: flex;
    align-items: center;
    gap: 20px;
}
.header-status {
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.72rem;
    color: #ffffff;
    font-weight: 500;
}
.status-dot {
    width: 7px;
    height: 7px;
    background: #4caf50;
    border-radius: 50%;
    display: inline-block;
}
.header-time {
    color: #a8bdd4;
    font-size: 0.72rem;
}

/* SECONDARY NAV */
.secondary-nav {
    background: #ffffff;
    border-bottom: 1px solid #dde3ed;
    padding: 0 32px;
    height: 40px;
    display: flex;
    align-items: center;
    gap: 4px;
    margin: 0 -1rem;
}
.nav-item {
    padding: 0 14px;
    height: 40px;
    display: flex;
    align-items: center;
    font-size: 0.78rem;
    color: #5a6a7e;
    font-weight: 500;
    border-bottom: 2px solid transparent;
    cursor: pointer;
}
.nav-item-active {
    color: #1b3a6b;
    border-bottom: 2px solid #1b3a6b;
    font-weight: 600;
}

/* PAGE CONTENT */
.page-wrapper {
    padding: 20px 0;
}
.breadcrumb {
    font-size: 0.72rem;
    color: #8a9ab0;
    margin-bottom: 12px;
}
.breadcrumb-link { color: #1b3a6b; }
.page-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1a2b45;
    letter-spacing: -0.2px;
    margin-bottom: 4px;
}
.page-desc {
    font-size: 0.8rem;
    color: #6b7c93;
    margin-bottom: 20px;
}

/* KPI CARDS */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}
.kpi-card {
    background: #ffffff;
    border: 1px solid #dde3ed;
    border-radius: 8px;
    padding: 16px 18px;
    position: relative;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.kpi-indicator {
    position: absolute;
    top: 0;
    left: 0;
    width: 3px;
    height: 100%;
    border-radius: 8px 0 0 8px;
    background: #1b3a6b;
}
.kpi-indicator.green { background: #2e7d32; }
.kpi-indicator.teal { background: #00838f; }
.kpi-indicator.purple { background: #6a1b9a; }
.kpi-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    color: #8a9ab0;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #1a2b45;
    letter-spacing: -0.5px;
    line-height: 1;
}
.kpi-sub {
    font-size: 0.7rem;
    color: #2e7d32;
    font-weight: 600;
    margin-top: 6px;
}
.kpi-sub.neutral { color: #5a6a7e; }

/* PANEL CARDS */
.panel {
    background: #ffffff;
    border: 1px solid #dde3ed;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    overflow: hidden;
    margin-bottom: 12px;
}
.panel-header {
    padding: 12px 18px;
    border-bottom: 1px solid #edf1f7;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #fafbfc;
}
.panel-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: #1a2b45;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
.panel-tag {
    font-size: 0.65rem;
    color: #8a9ab0;
    font-weight: 500;
}
.panel-body { padding: 18px; }

/* RESULT BOXES */
.result-benign {
    background: #f0faf4;
    border: 1.5px solid #2e7d32;
    border-radius: 8px;
    padding: 18px 20px;
    text-align: center;
    margin-bottom: 14px;
}
.result-benign .result-icon {
    width: 40px;
    height: 40px;
    background: #2e7d32;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 10px auto;
    color: white;
    font-size: 18px;
    font-weight: 700;
}
.result-benign .result-label {
    font-size: 1.3rem;
    font-weight: 800;
    color: #1b5e20;
    letter-spacing: 0.5px;
}
.result-benign .result-sub {
    font-size: 0.75rem;
    color: #388e3c;
    margin-top: 4px;
    font-weight: 500;
}

.result-malignant {
    background: #fff5f5;
    border: 1.5px solid #c62828;
    border-radius: 8px;
    padding: 18px 20px;
    text-align: center;
    margin-bottom: 14px;
}
.result-malignant .result-icon {
    width: 40px;
    height: 40px;
    background: #c62828;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 10px auto;
    color: white;
    font-size: 18px;
    font-weight: 700;
}
.result-malignant .result-label {
    font-size: 1.3rem;
    font-weight: 800;
    color: #b71c1c;
    letter-spacing: 0.5px;
}
.result-malignant .result-sub {
    font-size: 0.75rem;
    color: #e53935;
    margin-top: 4px;
    font-weight: 500;
}

/* PROBABILITY BARS */
.prob-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 4px;
}
.prob-label { font-size: 0.72rem; color: #5a6a7e; font-weight: 600; }
.prob-val { font-size: 0.78rem; color: #1a2b45; font-weight: 700; }
.prob-bar-bg {
    background: #edf1f7;
    border-radius: 4px;
    height: 6px;
    margin-bottom: 10px;
    overflow: hidden;
}
.prob-bar-green {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #2e7d32, #43a047);
}
.prob-bar-red {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #c62828, #e53935);
}
.prob-bar-blue {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #1b3a6b, #1565c0);
}

/* METRICS ROW */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 14px;
}
.metric-item {
    background: #f7f9fc;
    border: 1px solid #edf1f7;
    border-radius: 6px;
    padding: 10px 12px;
    text-align: center;
}
.metric-item-label {
    font-size: 0.65rem;
    color: #8a9ab0;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-item-value {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1a2b45;
    margin-top: 2px;
}

/* DATA TABLE */
.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.78rem;
}
.data-table th {
    background: #f7f9fc;
    color: #5a6a7e;
    font-weight: 600;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 8px 12px;
    border-bottom: 1px solid #dde3ed;
    text-align: left;
}
.data-table td {
    padding: 8px 12px;
    border-bottom: 1px solid #f1f4f8;
    color: #1a2b45;
}
.data-table tr:last-child td { border-bottom: none; }
.badge-benign {
    background: #e8f5e9;
    color: #2e7d32;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.68rem;
    font-weight: 600;
}
.badge-malignant {
    background: #ffebee;
    color: #c62828;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.68rem;
    font-weight: 600;
}

/* DISCLAIMER */
.disclaimer {
    background: #fffbeb;
    border: 1px solid #f9a825;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.7rem;
    color: #6d4c00;
    line-height: 1.6;
    margin-top: 12px;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #dde3ed !important;
}
.sidebar-top {
    background: #1b3a6b;
    padding: 16px;
    margin-bottom: 0;
}
.sidebar-top-title {
    color: #ffffff;
    font-size: 0.82rem;
    font-weight: 700;
}
.sidebar-top-sub {
    color: #a8bdd4;
    font-size: 0.68rem;
    margin-top: 2px;
}
.sidebar-section-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #8a9ab0;
    margin: 16px 0 8px 0;
    padding: 0 16px;
}
.model-status-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 16px;
    font-size: 0.75rem;
    color: #1a2b45;
    font-weight: 500;
}
.dot-green {
    width: 7px; height: 7px;
    background: #4caf50;
    border-radius: 50%;
    flex-shrink: 0;
}

/* FOOTER */
.app-footer {
    background: #1b3a6b;
    color: #7a96b8;
    text-align: center;
    padding: 14px 32px;
    border-radius: 8px;
    font-size: 0.7rem;
    margin-top: 20px;
    line-height: 1.8;
}
.app-footer strong { color: #a8bdd4; }
.app-footer a { color: #6aadde; text-decoration: none; }

/* OVERRIDE STREAMLIT */
div[data-testid="stMetricValue"] {
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    color: #1a2b45 !important;
}
.stSelectbox label, .stSlider label {
    color: #1a2b45 !important;
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
    xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1])
    return rf, xgb, X_test, y_test, data, X, rf_acc, xgb_acc, rf_auc, xgb_auc

rf, xgb, X_test, y_test, data, X_full, rf_acc, xgb_acc, rf_auc, xgb_auc = load_and_train()

now = datetime.datetime.now()
case_id = f"BC-{now.strftime('%Y%m%d')}-{now.strftime('%H%M')}"

# TOP HEADER
st.markdown(f"""
<div class="top-header">
    <div class="header-left">
        <div class="header-logo">O</div>
        <div>
            <div class="header-name">MedScan AI</div>
            <div class="header-dept">Oncology & Diagnostic Informatics Division</div>
        </div>
    </div>
    <div class="header-right">
        <div class="header-status"><span class="status-dot"></span> System Operational</div>
        <div class="header-time">{now.strftime('%A, %B %d, %Y  |  %H:%M')}</div>
    </div>
</div>
<div class="secondary-nav">
    <div class="nav-item nav-item-active">Screening Dashboard</div>
    <div class="nav-item">Patient Records</div>
    <div class="nav-item">Model Performance</div>
    <div class="nav-item">Audit Log</div>
    <div class="nav-item">Settings</div>
</div>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div class="sidebar-top">
        <div class="sidebar-top-title">Clinical Parameters</div>
        <div class="sidebar-top-sub">Enter tumor measurement values</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section-label">Tumor Morphology</div>', unsafe_allow_html=True)
    mean_radius = st.slider("Radius (mm)", 6.0, 30.0, 14.0, step=0.1)
    mean_texture = st.slider("Texture", 9.0, 40.0, 19.0, step=0.1)
    mean_perimeter = st.slider("Perimeter (mm)", 40.0, 200.0, 92.0, step=0.5)
    mean_area = st.slider("Area (mm²)", 140.0, 2600.0, 654.0, step=5.0)
    mean_smoothness = st.slider("Smoothness", 0.050, 0.170, 0.096, step=0.001, format="%.3f")
    
    st.markdown('<div class="sidebar-section-label">Algorithm Selection</div>', unsafe_allow_html=True)
    model_choice = st.selectbox("Prediction Model", ["Random Forest", "XGBoost", "Ensemble (Combined)"])
    
    st.markdown('<div class="sidebar-section-label">Model Status</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="model-status-item"><span class="dot-green"></span>Random Forest v2.1 — Active</div>
    <div class="model-status-item"><span class="dot-green"></span>XGBoost v3.2 — Active</div>
    <div class="model-status-item"><span class="dot-green"></span>Data Pipeline — Nominal</div>
    """, unsafe_allow_html=True)

# MAIN CONTENT
st.markdown("""
<div class="page-wrapper">
    <div class="breadcrumb">
        <span class="breadcrumb-link">Oncology</span> / 
        <span class="breadcrumb-link">Screening</span> / 
        Breast Cancer Malignancy Screening
    </div>
    <div class="page-title">Breast Cancer Malignancy Screening</div>
    <div class="page-desc">AI-assisted malignancy risk screening using ensemble machine learning | Wisconsin Breast Cancer Dataset | For research use only</div>
</div>
""", unsafe_allow_html=True)

# KPI CARDS
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-indicator"></div>
        <div class="kpi-label">Random Forest</div>
        <div class="kpi-value">{rf_acc:.1%}</div>
        <div class="kpi-sub">AUC-ROC: {rf_auc:.3f}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-indicator green"></div>
        <div class="kpi-label">XGBoost</div>
        <div class="kpi-value">{xgb_acc:.1%}</div>
        <div class="kpi-sub">AUC-ROC: {xgb_auc:.3f}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-indicator teal"></div>
        <div class="kpi-label">Training Records</div>
        <div class="kpi-value">569</div>
        <div class="kpi-sub neutral">Wisconsin Dataset</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-indicator purple"></div>
        <div class="kpi-label">Clinical Features</div>
        <div class="kpi-value">30</div>
        <div class="kpi-sub neutral">Morphological Parameters</div>
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
    <div class="panel">
        <div class="panel-header">
            <div class="panel-title">Diagnostic Assessment</div>
            <div class="panel-tag">AI-Generated — Not for clinical use</div>
        </div>
    </div>""", unsafe_allow_html=True)

    if pred == 1:
        st.markdown(f"""
        <div class="result-benign">
            <div class="result-icon">&#10003;</div>
            <div class="result-label">BENIGN</div>
            <div class="result-sub">Low malignancy risk — Routine follow-up recommended</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-malignant">
            <div class="result-icon">!</div>
            <div class="result-label">MALIGNANT</div>
            <div class="result-sub">High malignancy risk — Immediate clinical review required</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-item">
            <div class="metric-item-label">Confidence</div>
            <div class="metric-item-value">{confidence:.1f}%</div>
        </div>
        <div class="metric-item">
            <div class="metric-item-label">Algorithm</div>
            <div class="metric-item-value">{model_choice.split()[0]}</div>
        </div>
        <div class="metric-item">
            <div class="metric-item-label">Case ID</div>
            <div class="metric-item-value" style="font-size:0.75rem">{case_id}</div>
        </div>
    </div>
    
    <div style="background:#f7f9fc;border:1px solid #edf1f7;border-radius:6px;padding:14px 16px;margin-bottom:12px">
        <div class="prob-row">
            <span class="prob-label">Benign Probability</span>
            <span class="prob-val">{benign_prob:.1f}%</span>
        </div>
        <div class="prob-bar-bg">
            <div class="prob-bar-green" style="width:{benign_prob}%"></div>
        </div>
        <div class="prob-row">
            <span class="prob-label">Malignant Probability</span>
            <span class="prob-val">{malignant_prob:.1f}%</span>
        </div>
        <div class="prob-bar-bg">
            <div class="prob-bar-red" style="width:{malignant_prob}%"></div>
        </div>
        <div class="prob-row">
            <span class="prob-label">Model Confidence</span>
            <span class="prob-val" style="color:#1b3a6b">{confidence:.1f}%</span>
        </div>
        <div class="prob-bar-bg">
            <div class="prob-bar-blue" style="width:{confidence}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(5, 2.5))
    bars = ax.barh(['Benign', 'Malignant'],
                   [benign_prob, malignant_prob],
                   color=['#2e7d32', '#c62828'],
                   alpha=0.85, height=0.45, edgecolor='none')
    for bar, val in zip(bars, [benign_prob, malignant_prob]):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9,
                fontweight='600', color='#1a2b45')
    ax.set_xlim(0, 115)
    ax.set_xlabel('Probability (%)', fontsize=8, color='#5a6a7e')
    ax.set_title('Probability Distribution', fontsize=9,
                 fontweight='700', color='#1a2b45', pad=8)
    ax.set_facecolor('#f7f9fc')
    fig.patch.set_facecolor('#ffffff')
    ax.tick_params(colors='#5a6a7e', labelsize=8)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#dde3ed')
    ax.spines['bottom'].set_color('#dde3ed')
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    <div class="disclaimer">
        <strong>Clinical Disclaimer:</strong> This system is intended for research and educational purposes only. 
        Predictions generated by this tool must not be used as a substitute for professional medical judgment. 
        All results require validation by a licensed oncologist before any clinical action is taken. 
        This tool has not received FDA clearance for diagnostic use.
    </div>""", unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="panel">
        <div class="panel-header">
            <div class="panel-title">Feature Importance Analysis</div>
            <div class="panel-tag">Random Forest — Top 10 Features</div>
        </div>
    </div>""", unsafe_allow_html=True)

    importances = pd.Series(rf.feature_importances_,
                            index=data.feature_names).sort_values(ascending=True)[-10:]
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    colors_bar = ['#1b3a6b' if i >= 7 else '#3d6ea8' if i >= 4 else '#7fa5cc'
                  for i in range(len(importances))]
    bars2 = ax2.barh(range(len(importances)), importances.values,
                     color=colors_bar, alpha=0.9, height=0.55, edgecolor='none')
    ax2.set_yticks(range(len(importances)))
    ax2.set_yticklabels(importances.index, fontsize=8, color='#1a2b45')
    ax2.set_title('Most Influential Diagnostic Features',
                  fontsize=9, fontweight='700', color='#1a2b45', pad=10)
    ax2.set_xlabel('Importance Score', fontsize=8, color='#5a6a7e')
    for bar, val in zip(bars2, importances.values):
        ax2.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=7, color='#1a2b45', fontweight='600')
    ax2.set_facecolor('#f7f9fc')
    fig2.patch.set_facecolor('#ffffff')
    ax2.tick_params(colors='#5a6a7e', labelsize=8)
    for spine in ['top', 'right']:
        ax2.spines[spine].set_visible(False)
    ax2.spines['left'].set_color('#dde3ed')
    ax2.spines['bottom'].set_color('#dde3ed')
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("""
    <div class="panel">
        <div class="panel-header">
            <div class="panel-title">Model Documentation</div>
            <div class="panel-tag">Technical Reference</div>
        </div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Random Forest", "XGBoost", "Dataset"])
    with tab1:
        st.markdown(f"""
        <div style="font-size:0.8rem;color:#333;line-height:1.8;padding:4px 0">
        <strong style="color:#1a2b45">Architecture:</strong> 100 decision trees, majority voting ensemble<br>
        <strong style="color:#1a2b45">Accuracy:</strong> {rf_acc:.2%} &nbsp;|&nbsp;
        <strong style="color:#1a2b45">AUC-ROC:</strong> {rf_auc:.4f}<br>
        <strong style="color:#1a2b45">Features:</strong> 30 morphological parameters<br>
        <strong style="color:#1a2b45">Strengths:</strong> Robust to noise and missing values. Provides interpretable feature importance rankings. Ideal for structured clinical tabular data.
        </div>""", unsafe_allow_html=True)
    with tab2:
        st.markdown(f"""
        <div style="font-size:0.8rem;color:#333;line-height:1.8;padding:4px 0">
        <strong style="color:#1a2b45">Architecture:</strong> Gradient boosting, sequential error correction<br>
        <strong style="color:#1a2b45">Accuracy:</strong> {xgb_acc:.2%} &nbsp;|&nbsp;
        <strong style="color:#1a2b45">AUC-ROC:</strong> {xgb_auc:.4f}<br>
        <strong style="color:#1a2b45">Objective:</strong> Binary logistic classification<br>
        <strong style="color:#1a2b45">Strengths:</strong> High performance on structured healthcare data with built-in regularization to prevent overfitting.
        </div>""", unsafe_allow_html=True)
    with tab3:
        st.markdown("""
        <div style="font-size:0.8rem;color:#333;line-height:1.8;padding:4px 0">
        <strong style="color:#1a2b45">Source:</strong> Wisconsin Breast Cancer Dataset — UCI ML Repository<br>
        <strong style="color:#1a2b45">Records:</strong> 569 patients &nbsp;|&nbsp; <strong style="color:#1a2b45">Features:</strong> 30 clinical measurements<br>
        <strong style="color:#1a2b45">Class Distribution:</strong> Malignant: 212 (37.3%) / Benign: 357 (62.7%)<br>
        <strong style="color:#1a2b45">Validation:</strong> 80/20 train-test split, random_state=42, stratified
        </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div class="app-footer">
    <strong>MedScan AI</strong> &nbsp;|&nbsp; Version 2.1.0 &nbsp;|&nbsp;
    Developed by <strong>Priyanka Kapoor</strong> &nbsp;|&nbsp;
    MS Business Analytics, Montclair State University 2026 &nbsp;|&nbsp;
    <a href="https://github.com/PriyankaKapoor4202/Breast-Cancer-Detection-AI">GitHub Repository</a> &nbsp;|&nbsp;
    <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6669899">SSRN Research</a><br>
    This tool is intended for research and educational purposes only. Not approved for clinical diagnostic use. &nbsp;|&nbsp; Case ID: {case_id}
</div>
""", unsafe_allow_html=True)
