import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os

st.set_page_config(
    page_title="MataVision — City Planner Dashboard",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
      <div style="background:#7c3aed;border-radius:8px;width:32px;height:32px;
                  display:flex;align-items:center;justify-content:center">🛡️</div>
      <div>
        <div style="font-weight:700;font-size:.95rem">City Planner Dashboard</div>
        <div style="font-size:.72rem;color:#7c3aed">Infrastructure risk assessment for Salem, MA</div>
      </div>
    </div>""", unsafe_allow_html=True)
st.divider()

st.markdown("""
<div class="page-header">
  <h1>🏙️ City Planner Dashboard</h1>
  <p>Professional risk assessment for city infrastructure decisions — Salem, Massachusetts</p>
</div>""", unsafe_allow_html=True)

# ================================================================
# MONTH CONFIG — matches RF model Month feature exactly
# ================================================================
MONTH_CONFIG = {
    "All Months": {"month": None, "modifier": 0,   "season": "📅 Full Year"},
    "January":    {"month": 1,   "modifier": +10,  "season": "☃️ Winter"},
    "February":   {"month": 2,   "modifier": +8,   "season": "☃️ Winter"},
    "March":      {"month": 3,   "modifier": +2,   "season": "🌸 Spring"},
    "April":      {"month": 4,   "modifier": 0,    "season": "🌸 Spring"},
    "May":        {"month": 5,   "modifier": -2,   "season": "🌸 Spring"},
    "June":       {"month": 6,   "modifier": -4,   "season": "☀️ Summer"},
    "July":       {"month": 7,   "modifier": -5,   "season": "☀️ Summer"},
    "August":     {"month": 8,   "modifier": -3,   "season": "☀️ Summer"},
    "September":  {"month": 9,   "modifier": +2,   "season": "🍂 Fall"},
    "October":    {"month": 10,  "modifier": +12,  "season": "🍂 Fall"},
    "November":   {"month": 11,  "modifier": +5,   "season": "🍂 Fall"},
    "December":   {"month": 12,  "modifier": +8,   "season": "☃️ Winter"},
}
MONTH_NAMES = list(MONTH_CONFIG.keys())

MONTH_VOLUME = {
    None: 1.00,
    1: 1.05, 2: 1.03, 3: 0.90, 4: 0.85,
    5: 0.88, 6: 0.82, 7: 0.80, 8: 0.84,
    9: 0.92, 10: 1.20, 11: 1.05, 12: 1.08,
}

# ================================================================
# MASTER DATA
# ================================================================
ALL_HOUR_COUNTS = [12,8,6,5,4,10,22,55,78,60,55,62,70,65,68,95,110,108,80,55,38,28,22,15]
ALL_DAY_COUNTS  = {"Mon":420,"Tue":455,"Wed":470,"Thu":440,"Fri":520,"Sat":340,"Sun":290}

ALL_RISK = [
    {"Location":"Derby St / Washington Sq",   "Risk Score":89,"Risk Level":"High",  "Crash Count":142,"Peak Hour":"4-6 PM","Primary Factor":"High Volume + Wet Roads",  "Est. Budget":"$620K","Action":"Urgent Review",  "Road Type":"Highway"},
    {"Location":"North St / Essex St",         "Risk Score":76,"Risk Level":"High",  "Crash Count":98, "Peak Hour":"3-5 PM","Primary Factor":"Rush Hour Congestion",     "Est. Budget":"$480K","Action":"Urgent Review",  "Road Type":"Intersections"},
    {"Location":"Marlborough Rd / Ocean Ave",  "Risk Score":71,"Risk Level":"High",  "Crash Count":87, "Peak Hour":"5-7 PM","Primary Factor":"Nighttime Visibility",     "Est. Budget":"$380K","Action":"Schedule Review","Road Type":"Highway"},
    {"Location":"Bridge St / Federal St",      "Risk Score":68,"Risk Level":"Medium","Crash Count":74, "Peak Hour":"7-9 AM","Primary Factor":"AM Rush + Intersection",   "Est. Budget":"$310K","Action":"Schedule Review","Road Type":"Intersections"},
    {"Location":"Highland Ave / Jefferson Ave","Risk Score":62,"Risk Level":"Medium","Crash Count":61, "Peak Hour":"3-4 PM","Primary Factor":"School Zone Traffic",      "Est. Budget":"$260K","Action":"Schedule Review","Road Type":"Residential"},
    {"Location":"Lafayette St / Loring Ave",   "Risk Score":32,"Risk Level":"Low",   "Crash Count":28, "Peak Hour":"12-2 PM","Primary Factor":"Moderate Traffic",        "Est. Budget":"$100K","Action":"Monitor",        "Road Type":"Residential"},
    {"Location":"Canal St / Grove St",         "Risk Score":28,"Risk Level":"Low",   "Crash Count":19, "Peak Hour":"6-8 AM","Primary Factor":"Low Volume",               "Est. Budget":"$80K", "Action":"Monitor",        "Road Type":"Highway"},
]

# ================================================================
# FILTERS — on main page, no sidebar
# ================================================================
st.markdown("### 📍 Filter by Location & Time")
f1, f2, f3, f4 = st.columns([2, 2, 2, 1])
with f1:
    month_filter = st.selectbox("📅 Month", MONTH_NAMES)
with f2:
    road_filter  = st.selectbox("🛣️ Road Type", ["All Roads","Intersections","Highway","Residential"])
with f3:
    risk_filter  = st.selectbox("⚠️ Risk Level", ["All Levels","High Only","Medium & High","Low Only"])
with f4:
    show_rush    = st.toggle("Rush Hours", value=True)

# Download button inline
st.download_button("📥 Download Risk Report (CSV)",
    data="Location,Risk Score,Risk Level,Crash Count\nDerby St / Washington Sq,89,High,142\nNorth St / Essex St,76,High,98",
    file_name="salem_risk_report.csv", mime="text/csv")

st.divider()

# Unpack month config
month_cfg      = MONTH_CONFIG[month_filter]
selected_month = month_cfg["month"]
month_modifier = month_cfg["modifier"]
month_season   = month_cfg["season"]
vol_mult       = MONTH_VOLUME.get(selected_month, 1.0)

# Scale crash counts by month volume
hour_counts   = [int(round(c * vol_mult)) for c in ALL_HOUR_COUNTS]
day_data      = {d: int(round(c * vol_mult)) for d, c in ALL_DAY_COUNTS.items()}
days          = list(day_data.keys())
day_counts    = list(day_data.values())
total_crashes = int(round(3207 * vol_mult))
total_label   = f"{month_filter} Crashes (est.)" if month_filter != "All Months" else "Total Crashes Analyzed"

# Apply filters
risk_df = pd.DataFrame(ALL_RISK)
if road_filter != "All Roads":
    risk_df = risk_df[risk_df["Road Type"] == road_filter]
if risk_filter == "High Only":
    risk_df = risk_df[risk_df["Risk Level"] == "High"]
elif risk_filter == "Medium & High":
    risk_df = risk_df[risk_df["Risk Level"].isin(["High","Medium"])]
elif risk_filter == "Low Only":
    risk_df = risk_df[risk_df["Risk Level"] == "Low"]

risk_df = risk_df.copy()
risk_df["Risk Score"] = risk_df["Risk Score"].apply(
    lambda s: max(0, min(100, s + month_modifier)))

active_alerts = len(risk_df[risk_df["Risk Level"] == "High"])

# ================================================================
# MONTH CONTEXT BANNER
# ================================================================
m_bg  = "#fee2e2" if month_modifier >= 8 else "#fef3c7" if month_modifier >= 2 else "#f0fdf4"
m_bdr = "#dc2626" if month_modifier >= 8 else "#d97706" if month_modifier >= 2 else "#16a34a"
m_msg = ("High-risk month — winter conditions or Salem October surge"
         if month_modifier >= 8 else
         "Slightly elevated risk" if month_modifier >= 2 else
         "Lower-risk month — calmer driving conditions")

if month_filter != "All Months":
    st.markdown(f"""
    <div style="background:{m_bg};border:1px solid {m_bdr};border-radius:12px;
                padding:14px 20px;margin-bottom:20px;display:flex;
                justify-content:space-between;align-items:center">
      <div>
        <div style="font-weight:700;font-size:1rem">📅 {month_filter} · {month_season}</div>
        <div style="font-size:.85rem;color:#374151;margin-top:2px">{m_msg}</div>
      </div>
      <div style="text-align:right;font-size:.85rem">
        <div style="font-weight:700">Month risk adjustment: {'+' if month_modifier >= 0 else ''}{month_modifier} pts</div>
        <div style="color:#6b7280;font-size:.75rem">Based on Salem crash data patterns</div>
      </div>
    </div>""", unsafe_allow_html=True)

# ================================================================
# KPI CARDS
# ================================================================
st.markdown("### 📊 Risk Analysis Dashboard")
st.markdown(f"Key metrics — filtered to: **{month_filter}** · **{road_filter}** · **{risk_filter}**")

k1, k2, k3, k4 = st.columns(4)
k1.markdown(f"""<div class="planner-kpi">
  <span class="kpi-num">{total_crashes:,}</span>
  <div class="kpi-label">{total_label}</div>
  <div class="kpi-change kpi-down">Salem crash data 2020–present</div>
</div>""", unsafe_allow_html=True)

k2.markdown("""<div class="planner-kpi">
  <span class="kpi-num">71%</span>
  <div class="kpi-label">Best Model Accuracy</div>
  <div class="kpi-change">Tuned Random Forest</div>
</div>""", unsafe_allow_html=True)

k3.markdown(f"""<div class="planner-kpi">
  <span class="kpi-num">{active_alerts}</span>
  <div class="kpi-label">High-Risk Locations</div>
  <div class="kpi-change kpi-up">in current filter</div>
</div>""", unsafe_allow_html=True)

k4.markdown(f"""<div class="planner-kpi">
  <span class="kpi-num">{len(risk_df)}</span>
  <div class="kpi-label">Locations in View</div>
  <div class="kpi-change">after filters applied</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================================================================
# CRASH SEVERITY EXPLANATION
# ================================================================
with st.expander("📐 How is Crash Severity Measured?"):
    st.markdown("""
Crashes in Salem are grouped into three severity levels based on outcome:

| Level | What it means | Share of Salem crashes |
|---|---|---|
| **1 — Property Damage Only** | No injuries — vehicle or property damaged | 72.9% |
| **2 — Non-Fatal Injury** | At least one person injured, no deaths | 26.7% |
| **3 — Fatal Injury** | At least one fatality | 0.25% |

**Risk scores (0–100)** shown on this dashboard combine crash history at each location
with the time of day and month — the two strongest factors the model found in Salem data.
Locations scoring 70 or above are flagged for urgent infrastructure review.
    """)

# ================================================================
# CHARTS ROW 1
# ================================================================
c1, c2 = st.columns(2)

with c1:
    st.markdown(f"**🕐 Crash Frequency by Hour of Day — {month_filter}**")
    bar_colors = ["#dc2626" if c>=90 else "#f59e0b" if c>=55 else "#22c55e" for c in hour_counts]
    fig1 = go.Figure(go.Bar(x=list(range(24)), y=hour_counts, marker_color=bar_colors))
    if show_rush:
        fig1.add_vrect(x0=6.5, x1=9.5, fillcolor="rgba(251,146,60,.15)", line_width=0,
                       annotation_text="AM Rush", annotation_position="top left")
        fig1.add_vrect(x0=15.5, x1=18.5, fillcolor="rgba(239,68,68,.15)", line_width=0,
                       annotation_text="PM Rush", annotation_position="top left")
    fig1.update_layout(
        height=300, margin=dict(l=0,r=0,t=20,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Hour of Day", showgrid=False, tickvals=list(range(0,24,2))),
        yaxis=dict(title="Crashes", gridcolor="#f3f4f6"),
        showlegend=False
    )
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

with c2:
    st.markdown(f"**📅 Crash Frequency by Day of Week — {month_filter}**")
    day_colors = ["#dc2626" if d=="Fri" else "#2563eb" if d not in ["Sat","Sun"] else "#f59e0b"
                  for d in days]
    fig2 = go.Figure(go.Bar(x=days, y=day_counts, marker_color=day_colors,
                             text=day_counts, textposition="outside"))
    fig2.update_layout(
        height=300, margin=dict(l=0,r=0,t=20,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#f3f4f6"),
        showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# ================================================================
# CHARTS ROW 2
# ================================================================
c3, c4 = st.columns(2)

with c3:
    st.markdown("**📊 Crash Severity Distribution — Salem 2020–Present**")
    fig3 = go.Figure(go.Pie(
        labels=["Property Damage Only","Non-Fatal Injury","Fatal Injury"],
        values=[2343, 856, 8], hole=0.55,
        marker_colors=["#2563eb","#d97706","#dc2626"],
        textinfo="label+percent"
    ))
    fig3.update_layout(
        height=300, margin=dict(l=0,r=0,t=10,b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2, font=dict(size=11))
    )
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

with c4:
    st.markdown("**🗓️ Crash Volume by Month — Salem Crash Patterns**")
    months_list = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    month_vols  = [int(round(3207/12 * MONTH_VOLUME[m])) for m in range(1, 13)]
    m_colors    = ["#7c3aed" if (selected_month == i+1) else
                   "#dc2626" if MONTH_VOLUME[i+1] >= 1.10 else
                   "#22c55e" if MONTH_VOLUME[i+1] <= 0.85 else "#2563eb"
                   for i in range(12)]
    fig4 = go.Figure(go.Bar(
        x=months_list, y=month_vols,
        marker_color=m_colors,
        text=month_vols, textposition="outside"
    ))
    fig4.update_layout(
        height=300, margin=dict(l=0,r=0,t=20,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#f3f4f6"),
        showlegend=False
    )
    st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

# ================================================================
# HEATMAP
# ================================================================
st.markdown(f"**🌡️ Crash Density Heatmap — Hour vs Day of Week ({month_filter})**")
st.markdown('<div style="font-size:.78rem;color:#6b7280;margin-bottom:8px">PM rush hours (3–6 PM) show highest crash density — Salem data 2020 to present</div>',
            unsafe_allow_html=True)

day_names   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
hour_labels = ["12AM–2AM","2AM–4AM","4AM–6AM","6AM–8AM","8AM–10AM","10AM–12PM",
               "12PM–2PM","2PM–4PM","4PM–6PM","6PM–8PM","8PM–10PM","10PM–12AM"]

base_matrix = np.array([
    [8,  6,  4,  5,  6,  4,  3],
    [5,  4,  3,  4,  5,  6,  5],
    [8,  10, 9,  8,  11, 5,  4],
    [40, 45, 42, 38, 50, 20, 15],
    [60, 65, 62, 58, 70, 40, 35],
    [55, 58, 60, 55, 65, 50, 48],
    [65, 68, 70, 62, 75, 58, 52],
    [80, 85, 82, 78, 92, 62, 55],
    [95,100, 98, 90,115, 65, 55],
    [70, 72, 75, 68, 80, 72, 65],
    [40, 38, 42, 36, 48, 52, 48],
    [18, 16, 15, 14, 20, 24, 20]
])
heatmap_matrix = np.round(base_matrix * vol_mult).astype(int)

fig_hm = go.Figure(go.Heatmap(
    z=heatmap_matrix, x=day_names, y=hour_labels,
    colorscale="Reds", colorbar=dict(title="Crashes", thickness=12)
))
fig_hm.update_layout(
    height=420, margin=dict(l=80,r=20,t=10,b=20),
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(title="Day of Week"),
    yaxis=dict(title="Hour of Day", autorange="reversed", tickfont=dict(size=11))
)
st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})

# ================================================================
# RISK LOCATIONS TABLE
# ================================================================
st.markdown("### 📍 Priority Locations for Infrastructure Investment")

if len(risk_df) == 0:
    st.warning("No locations match the current filters.")
else:
    st.markdown(f'<div style="font-size:.82rem;color:#6b7280;margin-bottom:10px">Showing {len(risk_df)} of {len(ALL_RISK)} locations · Road: {road_filter} · Risk: {risk_filter} · Month: {month_filter}</div>',
                unsafe_allow_html=True)
    display_df = risk_df.drop(columns=["Road Type"]).reset_index(drop=True)

    def style_risk(val):
        if val=="High":   return "color:#dc2626;font-weight:bold"
        if val=="Medium": return "color:#d97706;font-weight:bold"
        return "color:#16a34a;font-weight:bold"

    def style_action(val):
        if val=="Urgent Review":   return "background:#fee2e2;color:#991b1b;font-weight:bold"
        if val=="Schedule Review": return "background:#fef3c7;color:#92400e;font-weight:bold"
        return "background:#dcfce7;color:#166534;font-weight:bold"

    styled = display_df.style \
        .map(style_risk,   subset=["Risk Level"]) \
        .map(style_action, subset=["Action"]) \
        .hide(axis="index")
    st.dataframe(styled, use_container_width=True, height=min(320, 60+len(risk_df)*45))

# ================================================================
# WHAT DRIVES RISK
# ================================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📊 What Drives Crash Risk in Salem?")
st.markdown('<div style="font-size:.85rem;color:#6b7280;margin-bottom:12px">Key factors that predict crash likelihood — from the MataVision model trained on Salem crash records. Helps prioritize where and when to focus infrastructure improvements.</div>',
            unsafe_allow_html=True)

features     = ["Time of Day", "Month of Year", "Weekend vs Weekday", "Rush Hour", "At an Intersection"]
importances  = [40, 30, 12, 10, 8]
descriptions = [
    "The hour of day is the strongest signal — late afternoon rush has the highest crash rate",
    "Crash rates shift by month — October peaks with Salem events, July is the calmest",
    "Weekdays see more crashes than weekends because of higher commuter traffic",
    "Morning (7–9 AM) and evening (4–6 PM) rush hours are the two most dangerous windows",
    "Intersections add some risk, though mid-block crashes are also common across Salem"
]
colors_fi = ["#7c3aed","#2563eb","#22c55e","#f59e0b","#9ca3af"]

fig_fi = go.Figure(go.Bar(
    y=features, x=importances, orientation="h",
    marker_color=colors_fi,
    text=[f"{i}%" for i in importances],
    textposition="outside",
    customdata=descriptions,
    hovertemplate="<b>%{y}</b><br>%{x}% of predictive power<br>%{customdata}<extra></extra>"
))
fig_fi.update_layout(
    height=280, margin=dict(l=0,r=50,t=10,b=0),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(range=[0,55], showgrid=True, gridcolor="#f3f4f6", ticksuffix="%"),
    yaxis=dict(showgrid=False),
    showlegend=False
)
st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})
st.caption("Time of day and month together account for 70% of what the model uses — meaning infrastructure improvements timed around peak hours and high-risk months will have the greatest impact.")

st.markdown("""<div class="footer">
  🛡️ <strong>MataVision</strong> · City Planner Dashboard · Salem, MA Capstone Project
</div>""", unsafe_allow_html=True)