import streamlit as st
from pathlib import Path

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="MataVision",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---- LOAD CSS ----
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---- HEADER / NAV BAR ----
st.markdown("""
<div class="navbar">
  <div class="navbar-brand">
    <div class="nav-logo">🛡️</div>
    <div>
      <div class="nav-title">MataVision</div>
      <div class="nav-sub">Predicting risks, protecting lives</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---- HERO ----
st.markdown("""
<div class="hero">
  <div class="hero-content">
    <h1 class="hero-h1">Safer Roads for<br><span class="hero-city">Salem, Massachusetts</span></h1>
    <p class="hero-desc">
      Machine learning predictions to prevent crashes and save lives.<br>
      <span class="hero-tagline">Predicting risks, protecting lives.</span>
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

# ---- STAT CARDS ----
st.markdown('<div class="hero-stats">', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
c1.markdown("""
<div class="stat-card">
  <span class="stat-num">3,207</span>
  <span class="stat-label">Crashes Analyzed</span>
</div>""", unsafe_allow_html=True)
c2.markdown("""
<div class="stat-card">
  <span class="stat-num">71%</span>
  <span class="stat-label">Best Model Accuracy</span>
</div>""", unsafe_allow_html=True)
c3.markdown("""
<div class="stat-card">
  <span class="stat-num">18%</span>
  <span class="stat-label">Risk Reduction</span>
</div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---- NAVIGATION BUTTONS ----
st.markdown('<div class="nav-btn-row">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("🗺️  Interactive Safety Map", use_container_width=True, type="primary"):
        st.switch_page("pages/1_Interactive_Safety_Map.py")

with col2:
    if st.button("🏙️  City Planner Dashboard", use_container_width=True):
        st.switch_page("pages/2_City_Planner_Dashboard.py")

with col3:
    if st.button("📋  Insurance Analytics", use_container_width=True):
        st.switch_page("pages/3_Insurance_Analytics.py")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<p style="text-align:center;color:#9ca3af;font-size:.85rem;margin-top:6px">Toggle between Community View and City Planner Mode on the Map page</p>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---- HOW IT WORKS LINK ----
col_l, col_c, col_r = st.columns([2, 1, 2])
with col_c:
    if st.button("↗  How It Works", use_container_width=True):
        st.switch_page("pages/4_How_It_Works.py")

st.markdown("---")

# ---- FEATURE CARDS ----
st.markdown('<div class="section-header"><h2>What MataVision Does</h2></div>', unsafe_allow_html=True)

fc1, fc2, fc3, fc4 = st.columns(4)
fc1.markdown("""
<div class="feature-card">
  <div class="feature-icon purple-icon">📍</div>
  <h4>Live Risk Mapping</h4>
  <p>Interactive map showing accident hotspots across Salem with ML-powered risk scores.</p>
</div>""", unsafe_allow_html=True)
fc2.markdown("""
<div class="feature-card">
  <div class="feature-icon blue-icon">🏙️</div>
  <h4>City Planner Mode</h4>
  <p>Professional dashboard with infrastructure planning insights and budget allocation recommendations.</p>
</div>""", unsafe_allow_html=True)
fc3.markdown("""
<div class="feature-card">
  <div class="feature-icon green-icon">📈</div>
  <h4>Temporal Predictions</h4>
  <p>Forecast crash risk by hour, day of week, and season using Random Forest and XGBoost.</p>
</div>""", unsafe_allow_html=True)
fc4.markdown("""
<div class="feature-card">
  <div class="feature-icon red-icon">🔔</div>
  <h4>Community Alerts</h4>
  <p>Safety tips and alerts tailored for Salem residents based on real crash patterns.</p>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---- ML MODELS SECTION ----
st.markdown('<div class="section-header"><h2>ML Models Powering Predictions</h2><p>Trained on official Salem crash records</p></div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

m1.markdown("""
<div class="model-card best-model">
  <div class="model-badge">Best Performer</div>
  <h4>Tuned Random Forest</h4>
  <div class="model-acc">71%</div>
  <p>GridSearchCV tuned (200 estimators, max_depth=10). Your best accuracy across all models.</p>
  <div class="model-bar"><div class="model-fill" style="width:71%"></div></div>
</div>""", unsafe_allow_html=True)

m2.markdown("""
<div class="model-card">
  <h4>RF Cross-Validation</h4>
  <div class="model-acc">67%</div>
  <p>5-fold stratified cross-validation average. Macro F1 ~0.31 reflects class imbalance challenge.</p>
  <div class="model-bar"><div class="model-fill" style="width:67%"></div></div>
</div>""", unsafe_allow_html=True)

m3.markdown("""
<div class="model-card">
  <h4>Random Forest (base)</h4>
  <div class="model-acc">56%</div>
  <p>300 estimators, balanced_subsample. Stronger than logistic regression on minority classes.</p>
  <div class="model-bar"><div class="model-fill" style="width:56%"></div></div>
</div>""", unsafe_allow_html=True)

m4.markdown("""
<div class="model-card">
  <h4>Logistic Regression</h4>
  <div class="model-acc">33–39%</div>
  <p>Baseline model. SMOTE improved recall but overall accuracy remained limited.</p>
  <div class="model-bar"><div class="model-fill" style="width:39%"></div></div>
</div>""", unsafe_allow_html=True)

# ---- FOOTER ----
st.markdown("""
<div class="footer">
  🛡️ <strong>MataVision</strong> · Salem, Massachusetts · Machine Learning for Road Safety · Capstone Project
</div>
""", unsafe_allow_html=True)