import streamlit as st
import plotly.graph_objects as go
import os

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="MataVision — How It Works",
    page_icon="↗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---- CSS ----
css_path = os.path.join(os.path.dirname(__file__), '..', 'styles.css')
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---- NAV ----
col_back, col_brand = st.columns([1, 5])
with col_back:
    if st.button("← Back", key="back"):
        st.switch_page("app.py")
with col_brand:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:8px 0">
      <div style="background:#7c3aed;border-radius:8px;width:32px;height:32px;display:flex;align-items:center;justify-content:center">🛡️</div>
      <div>
        <div style="font-weight:700;font-size:.95rem">How It Works</div>
        <div style="font-size:.72rem;color:#7c3aed">Predicting risks, protecting lives</div>
      </div>
    </div>""", unsafe_allow_html=True)
st.divider()

st.markdown("""
<div class="page-header">
  <h1>↗ How MataVision Works</h1>
  <p>A walkthrough of how Salem crash data becomes actionable safety predictions</p>
</div>""", unsafe_allow_html=True)

# ---- PIPELINE STEPS ----
steps = [
    ("1", "Data Collection — Salem Crash Data",
     "Official Salem, MA crash records containing thousands of incidents. Each record includes crash date/time, location (X/Y coordinates), road type, intersection status, weather conditions, severity, and more. Data is loaded from a structured CSV and cleaned to remove noise and incomplete records."),
    ("2", "Data Cleaning & Feature Engineering",
     "Columns with over 90% missing values are dropped. Forward-fill handles smaller gaps. Key features are engineered: <b>Is_Rush_Hour</b> (7–9 AM or 4–6 PM), <b>Is_Weekend</b>, <b>At_Intersection</b>, <b>Hour</b>, <b>Month</b>, and <b>Season</b>. These capture the core temporal and contextual patterns that drive crash risk."),
    ("3", "Crash Severity Encoding",
     "Severity is mapped to numeric scores: <b>1 = Property Damage Only</b>, <b>2 = Non-Fatal Injury</b>, <b>3 = Fatal Injury</b>. The dataset is heavily imbalanced — 72.9% are Property Damage cases while fatal crashes are extremely rare (0.25%). This is addressed using SMOTE oversampling and class weighting."),
    ("4", "Model Training & Evaluation",
     "Multiple models were trained on 80% of the data and evaluated on the remaining 20%. The best performer was <b>Random Forest</b> (300 estimators, balanced_subsample class weights) achieving 56% accuracy — significantly better than predicting the majority class. XGBoost achieved ~60%. SMOTE improved minority class recall across all models."),
    ("5", "Risk Scoring & Hotspot Detection",
     "Model predictions are converted into risk scores (0–100) per intersection. Locations with 70%+ predicted crash probability are flagged as High Risk (red), 40–69% as Medium Risk (yellow), and below 40% as Low Risk (green). Scores update dynamically when time-of-day or weather filters are applied."),
    ("6", "Interactive Dashboard & Deployment",
     "Results are presented in two views: <b>Community Mode</b> shows accessible safety tips and color-coded risk zones for everyday drivers. <b>Planner Mode</b> provides professional-grade tables with exact risk scores, recommended actions, and budget insights for city infrastructure teams.")
]

for num, title, desc in steps:
    st.markdown(f"""
    <div class="step-card">
      <div class="step-num">{num}</div>
      <div class="step-body">
        <h3>{title}</h3>
        <p>{desc}</p>
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---- KEY FINDINGS ----
st.markdown("### 📌 Key Findings from Salem Data")

kf1, kf2, kf3, kf4 = st.columns(4)
kf1.markdown("""
<div style="background:white;border:1px solid #e5e7eb;border-radius:12px;padding:18px">
  <div style="color:#7c3aed;font-weight:700;font-size:.9rem;margin-bottom:10px">🕐 Temporal Patterns</div>
  <ul style="font-size:.83rem;color:#6b7280;line-height:2;padding-left:14px;margin:0">
    <li>Crashes peak <b>3–6 PM</b> (rush hour)</li>
    <li>Friday has highest crash count</li>
    <li>Weekends 30% lower than weekdays</li>
    <li>Minimum crashes 2–5 AM</li>
  </ul>
</div>""", unsafe_allow_html=True)
kf2.markdown("""
<div style="background:white;border:1px solid #e5e7eb;border-radius:12px;padding:18px">
  <div style="color:#2563eb;font-weight:700;font-size:.9rem;margin-bottom:10px">📍 Spatial Patterns</div>
  <ul style="font-size:.83rem;color:#6b7280;line-height:2;padding-left:14px;margin:0">
    <li>Clusters along <b>major corridors</b></li>
    <li>5 intersections highest risk</li>
    <li>Most crashes NOT at intersections</li>
    <li>Fatal crashes dispersed, not clustered</li>
  </ul>
</div>""", unsafe_allow_html=True)
kf3.markdown("""
<div style="background:white;border:1px solid #e5e7eb;border-radius:12px;padding:18px">
  <div style="color:#16a34a;font-weight:700;font-size:.9rem;margin-bottom:10px">🌧️ Condition Patterns</div>
  <ul style="font-size:.83rem;color:#6b7280;line-height:2;padding-left:14px;margin:0">
    <li><b>34%</b> occur on wet/rainy roads</li>
    <li><b>29%</b> happen after dark</li>
    <li>Winter months elevated rates</li>
    <li>Fog correlates with severity</li>
  </ul>
</div>""", unsafe_allow_html=True)
kf4.markdown("""
<div style="background:white;border:1px solid #e5e7eb;border-radius:12px;padding:18px">
  <div style="color:#dc2626;font-weight:700;font-size:.9rem;margin-bottom:10px">🤖 ML Insights</div>
  <ul style="font-size:.83rem;color:#6b7280;line-height:2;padding-left:14px;margin:0">
    <li>Hour of Day <b>#1 predictor</b> (82%)</li>
    <li>Intersection status ranks 2nd (74%)</li>
    <li>SMOTE improved recall by ~15%</li>
    <li>Random Forest beats Logistic Reg.</li>
  </ul>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---- SEVERITY DIST CHART ----
st.markdown("### 📊 Severity Class Distribution")
st.markdown('<div style="font-size:.85rem;color:#6b7280;margin-bottom:10px">The class imbalance challenge — why accuracy alone is not a sufficient metric</div>', unsafe_allow_html=True)

fig = go.Figure(go.Bar(
    y=['Property Damage Only (Sev. 1)', 'Non-Fatal Injury (Sev. 2)', 'Fatal Injury (Sev. 3)'],
    x=[2343, 856, 8],
    orientation='h',
    marker_color=['#2563eb', '#d97706', '#dc2626'],
    text=['2,343 (73.1%)', '856 (26.7%)', '8 (0.25%)'],
    textposition='outside'
))
fig.update_layout(
    height=240, margin=dict(l=0,r=100,t=10,b=0),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(showgrid=True, gridcolor='#f3f4f6'),
    yaxis=dict(showgrid=False),
    showlegend=False
)
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ---- TECH STACK ----
st.markdown("### 🛠️ Technology Stack")
st.markdown("""
<div style="background:#f5f3ff;border-radius:14px;padding:20px">
  <span class="tech-tag">Python (Google Colab)</span>
  <span class="tech-tag">Pandas &amp; NumPy</span>
  <span class="tech-tag">scikit-learn</span>
  <span class="tech-tag">XGBoost</span>
  <span class="tech-tag">SMOTE (imbalanced-learn)</span>
  <span class="tech-tag">Random Forest</span>
  <span class="tech-tag">MLP Neural Network</span>
  <span class="tech-tag">SVM</span>
  <span class="tech-tag">Streamlit</span>
  <span class="tech-tag">Folium (Maps)</span>
  <span class="tech-tag">Plotly</span>
  <span class="tech-tag">GitHub</span>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---- FOOTER ----
st.markdown("""
<div class="footer">
  🛡️ <strong>MataVision</strong> · How It Works · Salem, MA Capstone Project
</div>""", unsafe_allow_html=True)