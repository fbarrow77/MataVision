import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os

st.set_page_config(
    page_title="MataVision — Insurance Analytics",
    page_icon="📋",
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
        <div style="font-weight:700;font-size:.95rem">Insurance Analytics</div>
        <div style="font-size:.72rem;color:#7c3aed">Location-based crash risk for underwriting decisions</div>
      </div>
    </div>""", unsafe_allow_html=True)
st.divider()

st.markdown("""
<div class="page-header">
  <h1>📋 Insurance Risk Analytics</h1>
  <p>Review crash severity patterns by client location and time of year — Salem, MA</p>
</div>""", unsafe_allow_html=True)

# ================================================================
# MONTH CONFIG — matches RF model Month feature exactly
# ================================================================
MONTH_CONFIG = {
    "January":   {"month": 1,  "modifier": +10, "season": "☃️ Winter"},
    "February":  {"month": 2,  "modifier": +8,  "season": "☃️ Winter"},
    "March":     {"month": 3,  "modifier": +2,  "season": "🌸 Spring"},
    "April":     {"month": 4,  "modifier": 0,   "season": "🌸 Spring"},
    "May":       {"month": 5,  "modifier": -2,  "season": "🌸 Spring"},
    "June":      {"month": 6,  "modifier": -4,  "season": "☀️ Summer"},
    "July":      {"month": 7,  "modifier": -5,  "season": "☀️ Summer"},
    "August":    {"month": 8,  "modifier": -3,  "season": "☀️ Summer"},
    "September": {"month": 9,  "modifier": +2,  "season": "🍂 Fall"},
    "October":   {"month": 10, "modifier": +12, "season": "🍂 Fall"},
    "November":  {"month": 11, "modifier": +5,  "season": "🍂 Fall"},
    "December":  {"month": 12, "modifier": +8,  "season": "☃️ Winter"},
}
MONTH_NAMES = list(MONTH_CONFIG.keys())

# ================================================================
# LOCATION RISK DATA
# ================================================================
LOCATION_DATA = {
    "Derby St / Washington Sq": {
        "base_risk": 89, "crashes": 142, "avg_severity": 1.8,
        "peak_hour": "4-6 PM", "wet_crashes_pct": 38, "night_crashes_pct": 31,
        "zone": "Downtown", "premium_mod": "+22%",
        "desc": "Highest-risk zone. Heavy foot traffic, tourist area, multiple turning conflicts."
    },
    "North St / Essex St": {
        "base_risk": 76, "crashes": 98, "avg_severity": 1.7,
        "peak_hour": "3-5 PM", "wet_crashes_pct": 35, "night_crashes_pct": 28,
        "zone": "Downtown", "premium_mod": "+18%",
        "desc": "Major downtown intersection with high PM rush hour congestion."
    },
    "Marlborough Rd / Ocean Ave": {
        "base_risk": 71, "crashes": 87, "avg_severity": 1.9,
        "peak_hour": "5-7 PM", "wet_crashes_pct": 42, "night_crashes_pct": 38,
        "zone": "Coastal", "premium_mod": "+16%",
        "desc": "Coastal road with poor nighttime visibility and frequent wet conditions."
    },
    "Bridge St / Federal St": {
        "base_risk": 68, "crashes": 74, "avg_severity": 1.6,
        "peak_hour": "7-9 AM", "wet_crashes_pct": 30, "night_crashes_pct": 22,
        "zone": "North Salem", "premium_mod": "+14%",
        "desc": "AM rush hour bottleneck connecting residential north Salem to downtown."
    },
    "Highland Ave / Jefferson Ave": {
        "base_risk": 62, "crashes": 61, "avg_severity": 1.5,
        "peak_hour": "3-4 PM", "wet_crashes_pct": 28, "night_crashes_pct": 20,
        "zone": "Residential", "premium_mod": "+10%",
        "desc": "School zone traffic. Risk spikes 3-4 PM on weekdays."
    },
    "Lafayette St / Loring Ave": {
        "base_risk": 32, "crashes": 28, "avg_severity": 1.2,
        "peak_hour": "12-2 PM", "wet_crashes_pct": 18, "night_crashes_pct": 12,
        "zone": "South Salem", "premium_mod": "-5%",
        "desc": "Lower-risk residential area. Mostly minor property damage incidents."
    },
    "Canal St / Grove St": {
        "base_risk": 28, "crashes": 19, "avg_severity": 1.1,
        "peak_hour": "6-8 AM", "wet_crashes_pct": 15, "night_crashes_pct": 10,
        "zone": "South Salem", "premium_mod": "-8%",
        "desc": "Lowest risk zone. Low traffic volume, mostly early morning incidents."
    },
    "Salem Common / Washington Sq": {
        "base_risk": 40, "crashes": 35, "avg_severity": 1.3,
        "peak_hour": "2-4 PM", "wet_crashes_pct": 22, "night_crashes_pct": 15,
        "zone": "Downtown", "premium_mod": "+2%",
        "desc": "Moderate risk near the common. Weekend tourist traffic elevates risk."
    },
    "Peabody St / North St": {
        "base_risk": 48, "crashes": 44, "avg_severity": 1.4,
        "peak_hour": "8-9 AM", "wet_crashes_pct": 25, "night_crashes_pct": 18,
        "zone": "North Salem", "premium_mod": "+6%",
        "desc": "Commuter corridor. Morning rush hour is the primary risk window."
    },
    "Winter St / Essex St": {
        "base_risk": 45, "crashes": 40, "avg_severity": 1.4,
        "peak_hour": "5-6 PM", "wet_crashes_pct": 24, "night_crashes_pct": 20,
        "zone": "Downtown", "premium_mod": "+5%",
        "desc": "Evening commute traffic. Winter months significantly raise risk."
    },
}

def adjusted_risk(location, month_modifier):
    base = LOCATION_DATA[location]["base_risk"]
    return max(0, min(100, base + month_modifier))

def risk_label(score):
    if score >= 70:  return "High Risk",     "#dc2626"
    if score >= 40:  return "Moderate Risk", "#f59e0b"
    return "Low Risk", "#16a34a"

# ================================================================
# CLIENT PROFILE FILTERS — on main page, no sidebar
# ================================================================
st.markdown("### 👤 Client Profile")
f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
with f1:
    client_location = st.selectbox(
        "📍 Client Location",
        list(LOCATION_DATA.keys()),
        help="Select the intersection nearest to where the client lives or drives most"
    )
with f2:
    client_month_name = st.selectbox("📅 Month", MONTH_NAMES)
with f3:
    client_type = st.selectbox("🚗 Policy Type", [
        "Personal Auto", "Commercial Vehicle", "Fleet Policy"
    ])
with f4:
    years_driving = st.slider("📅 Years Driving in Salem", 0, 20, 3)

st.divider()

# Unpack month config
month_cfg      = MONTH_CONFIG[client_month_name]
client_month   = month_cfg["month"]
month_modifier = month_cfg["modifier"]
month_season   = month_cfg["season"]

# ================================================================
# COMPUTE CLIENT RISK PROFILE
# ================================================================
loc_data   = LOCATION_DATA[client_location]
curr_score = adjusted_risk(client_location, month_modifier)
base_score = loc_data["base_risk"]
curr_lbl, curr_col = risk_label(curr_score)

# ================================================================
# PREMIUM CALCULATION
# Base: $1,200 — Massachusetts average liability-only auto premium
# Source: The Zebra, MA auto insurance data ($1,201/yr)
# ================================================================
base_premium        = 1_200
loc_mod_pct         = int(loc_data["premium_mod"].replace("%","").replace("+",""))
loc_adj             = int(base_premium * loc_mod_pct / 100)
severity_surcharge  = int(max(0, (loc_data["avg_severity"] - 1.0) * 150))
policy_mods         = {"Personal Auto": 0, "Commercial Vehicle": 350, "Fleet Policy": 600}
policy_adj          = policy_mods[client_type]
experience_discount = min(years_driving * 25, 250)
month_surcharge     = max(0, month_modifier * 4)  # each risk point = $4

est_premium = (base_premium + loc_adj + severity_surcharge +
               policy_adj + month_surcharge - experience_discount)
est_premium = max(800, est_premium)

# ================================================================
# CLIENT RISK SUMMARY
# ================================================================
st.markdown(f"### Risk Profile: **{client_location}**")
st.markdown(f"*{loc_data['zone']} zone · {client_type} · {years_driving} years driving · {client_month_name} ({month_season})*")
st.markdown("<br>", unsafe_allow_html=True)

# Month context banner
m_bg  = "#fee2e2" if month_modifier >= 8 else "#fef3c7" if month_modifier >= 2 else "#f0fdf4"
m_bdr = "#dc2626" if month_modifier >= 8 else "#d97706" if month_modifier >= 2 else "#16a34a"
m_msg = ("⚠️ High-risk month — winter conditions or Salem October surge"
         if month_modifier >= 8 else
         "⚠️ Slightly elevated risk this time of year"
         if month_modifier >= 2 else
         "✅ Lower-risk time of year — good driving conditions")
st.markdown(f"""
<div style="background:{m_bg};border:1px solid {m_bdr};border-radius:10px;
            padding:12px 18px;margin-bottom:16px;font-size:.85rem">
  <b>📅 {client_month_name} ({month_season})</b> — {m_msg}
  <span style="color:#6b7280;font-size:.78rem;margin-left:8px">
    Month risk adjustment: {'+' if month_modifier >= 0 else ''}{month_modifier} pts
    · Based on Salem crash data patterns
  </span>
</div>""", unsafe_allow_html=True)

# Risk score cards
r1, r2, r3, r4 = st.columns(4)

r1.markdown(f"""
<div style="background:white;border:2px solid {curr_col};border-radius:14px;padding:18px;text-align:center">
  <div style="font-size:.75rem;color:#9ca3af;margin-bottom:4px">CURRENT RISK SCORE</div>
  <div style="font-family:'Syne',sans-serif;font-size:2.5rem;font-weight:800;color:{curr_col}">{curr_score}</div>
  <div style="font-size:.8rem;font-weight:600;color:{curr_col}">{curr_lbl}</div>
  <div style="font-size:.7rem;color:#9ca3af;margin-top:4px">location + month adjustment</div>
</div>""", unsafe_allow_html=True)

r2.markdown(f"""
<div style="background:white;border:1px solid #e5e7eb;border-radius:14px;padding:18px;text-align:center">
  <div style="font-size:.75rem;color:#9ca3af;margin-bottom:4px">BASE LOCATION RISK</div>
  <div style="font-family:'Syne',sans-serif;font-size:2.5rem;font-weight:800;color:#7c3aed">{base_score}</div>
  <div style="font-size:.8rem;color:#6b7280">Historical average</div>
  <div style="font-size:.7rem;color:#9ca3af;margin-top:4px">{loc_data['crashes']} recorded crashes</div>
</div>""", unsafe_allow_html=True)

r3.markdown(f"""
<div style="background:white;border:1px solid #e5e7eb;border-radius:14px;padding:18px;text-align:center">
  <div style="font-size:.75rem;color:#9ca3af;margin-bottom:4px">EST. ANNUAL PREMIUM</div>
  <div style="font-family:'Syne',sans-serif;font-size:2.5rem;font-weight:800;color:#2563eb">${est_premium:,}</div>
  <div style="font-size:.8rem;color:#6b7280">Location adj: {loc_data['premium_mod']} · {client_type}</div>
  <div style="font-size:.7rem;color:#9ca3af;margin-top:4px">Exp. discount: -${min(years_driving*25,250):,}</div>
</div>""", unsafe_allow_html=True)

r4.markdown(f"""
<div style="background:white;border:1px solid #e5e7eb;border-radius:14px;padding:18px;text-align:center">
  <div style="font-size:.75rem;color:#9ca3af;margin-bottom:4px">AVG CRASH SEVERITY</div>
  <div style="font-family:'Syne',sans-serif;font-size:2.5rem;font-weight:800;color:#d97706">{loc_data['avg_severity']}</div>
  <div style="font-size:.8rem;color:#6b7280">out of 3.0</div>
  <div style="font-size:.7rem;color:#9ca3af;margin-top:4px">1=Property · 2=Injury · 3=Fatal</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Premium breakdown expander
with st.expander("💰 How is the premium calculated? Click to see full breakdown"):
    st.markdown(f"""
| Component | Amount | Reason |
|---|---|---|
| **Base Premium** | $1,200 | Massachusetts avg liability-only auto policy (The Zebra: $1,201/yr) |
| **Location Adjustment ({loc_data['premium_mod']})** | ${loc_adj:+,} | {client_location} has {loc_data['crashes']} recorded crashes |
| **Severity Surcharge** | +${severity_surcharge} | Avg crash severity {loc_data['avg_severity']}/3.0 — each 0.1 above 1.0 = +$15 |
| **Month Risk Surcharge** | +${month_surcharge} | {client_month_name} — {'+' if month_modifier >= 0 else ''}{month_modifier} pt adjustment × $4/pt |
| **Policy Type** | +${policy_adj} | {client_type} |
| **Experience Discount** | -${min(years_driving*25,250):,} | {years_driving} years driving in Salem — $25/yr (max -$250) |
| **Estimated Annual Premium** | **${est_premium:,}** | Minimum floor: $800 |

> *This is an ML-informed estimate based on Salem crash data patterns.
> Actual premiums are set by licensed insurers and depend on additional factors
> including driving record, vehicle type, and credit score.*
    """)

# Location description
st.markdown(f"""
<div style="background:#f8fafc;border-left:4px solid {curr_col};border-radius:8px;padding:14px 18px;margin-bottom:16px">
  <div style="font-weight:700;font-size:.9rem;margin-bottom:4px">📍 Location Risk Summary</div>
  <div style="font-size:.85rem;color:#374151">{loc_data['desc']}</div>
  <div style="font-size:.78rem;color:#6b7280;margin-top:6px">
    Peak crash hour: <b>{loc_data['peak_hour']}</b> ·
    Wet road crashes: <b>{loc_data['wet_crashes_pct']}%</b> ·
    Night crashes: <b>{loc_data['night_crashes_pct']}%</b>
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================================================================
# RISK BY HOUR — for this location
# ================================================================
st.markdown("### 📊 Crash Severity Patterns for This Location")
c1, c2 = st.columns(2)

with c1:
    st.markdown(f"**Risk Score by Hour of Day — {client_location.split('/')[0].strip()}**")
    hours = list(range(24))
    hour_mods = {h: (+18 if 16<=h<=18 else +12 if 7<=h<=9 else
                     +8 if h>=22 or h<=5 else -5 if 10<=h<=14 else 0)
                 for h in hours}
    hourly_scores = [max(0, min(100, base_score + hour_mods[h] + month_modifier))
                     for h in hours]
    bar_colors = ["#dc2626" if s>=70 else "#f59e0b" if s>=40 else "#22c55e"
                  for s in hourly_scores]
    fig_h = go.Figure(go.Bar(x=hours, y=hourly_scores, marker_color=bar_colors))
    fig_h.add_hline(y=70, line_dash="dot", line_color="#dc2626",
                    annotation_text="High Risk threshold")
    fig_h.add_hline(y=40, line_dash="dot", line_color="#f59e0b",
                    annotation_text="Moderate threshold")
    fig_h.update_layout(
        height=300, margin=dict(l=0,r=0,t=20,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Hour of Day", showgrid=False, tickvals=list(range(0,24,2))),
        yaxis=dict(title="Risk Score", range=[0,105], gridcolor="#f3f4f6"),
        showlegend=False
    )
    st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar": False})

with c2:
    st.markdown("**Crash Cause Breakdown — This Location**")
    wet_pct   = loc_data["wet_crashes_pct"]
    night_pct = loc_data["night_crashes_pct"]
    other_pct = 100 - wet_pct - night_pct
    fig_pie = go.Figure(go.Pie(
        labels=["Wet / Rainy Roads", "Night / Low Visibility", "Other Conditions"],
        values=[wet_pct, night_pct, other_pct],
        hole=0.55,
        marker_colors=["#2563eb", "#7c3aed", "#9ca3af"],
        textinfo="label+percent"
    ))
    fig_pie.update_layout(
        height=300, margin=dict(l=0,r=0,t=10,b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2, font=dict(size=10))
    )
    st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

# ================================================================
# COMPARE ALL LOCATIONS
# ================================================================
st.markdown("### 📍 Location Risk Comparison — All Salem Zones")
st.markdown(f'<div style="font-size:.82rem;color:#6b7280;margin-bottom:12px">All locations adjusted for {client_month_name} ({month_season}) conditions</div>',
            unsafe_allow_html=True)

comp_data = []
for loc, d in LOCATION_DATA.items():
    score = adjusted_risk(loc, month_modifier)
    lbl, col = risk_label(score)
    comp_data.append({
        "Location": loc, "Zone": d["zone"],
        "Risk Score": score,
        "Base Risk": d["base_risk"],
        "Risk Level": lbl,
        "Crashes": d["crashes"],
        "Peak Hour": d["peak_hour"],
        "Premium Adjustment": d["premium_mod"],
        "Selected": "★ YOUR CLIENT" if loc == client_location else ""
    })

comp_df = pd.DataFrame(comp_data).sort_values("Risk Score", ascending=False)

bar_cols = ["#7c3aed" if r == client_location else
            "#dc2626" if s >= 70 else "#f59e0b" if s >= 40 else "#22c55e"
            for r, s in zip(comp_df["Location"], comp_df["Risk Score"])]

fig_comp = go.Figure(go.Bar(
    x=[l.split("/")[0].strip() for l in comp_df["Location"]],
    y=comp_df["Risk Score"],
    marker_color=bar_cols,
    text=comp_df["Risk Score"],
    textposition="outside",
    customdata=comp_df["Location"],
    hovertemplate="<b>%{customdata}</b><br>Risk Score: %{y}<extra></extra>"
))
fig_comp.add_hline(y=70, line_dash="dash", line_color="#dc2626",
                   annotation_text="High Risk threshold")
fig_comp.update_layout(
    height=320, margin=dict(l=0,r=0,t=20,b=0),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=False, tickfont=dict(size=10)),
    yaxis=dict(range=[0,115], gridcolor="#f3f4f6"),
    showlegend=False
)
st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})

def style_risk_lbl(val):
    if "High" in val:     return "color:#dc2626;font-weight:bold"
    if "Moderate" in val: return "color:#d97706;font-weight:bold"
    return "color:#16a34a;font-weight:bold"

def style_selected(val):
    if val == "★ YOUR CLIENT": return "background:#f5f3ff;color:#7c3aed;font-weight:bold"
    return ""

styled_comp = comp_df[["Location","Zone","Risk Score","Risk Level",
                        "Crashes","Peak Hour","Premium Adjustment","Selected"]].style \
    .map(style_risk_lbl, subset=["Risk Level"]) \
    .map(style_selected, subset=["Selected"]) \
    .hide(axis="index")
st.dataframe(styled_comp, use_container_width=True, height=380)

st.markdown("""<div class="footer">
  🛡️ <strong>MataVision</strong> · Insurance Analytics · Salem, MA Capstone Project
</div>""", unsafe_allow_html=True)