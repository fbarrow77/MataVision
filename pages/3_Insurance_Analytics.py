import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import urllib.request
import json

st.set_page_config(
    page_title="MataVision — Insurance Analytics",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
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
  <p>Review crash severity patterns by client location, weather, and time of day — Salem, MA</p>
</div>""", unsafe_allow_html=True)

# ================================================================
# LIVE WEATHER
# ================================================================
@st.cache_data(ttl=1800)
def get_salem_weather():
    try:
        url = ("https://api.open-meteo.com/v1/forecast"
               "?latitude=42.519&longitude=-70.896"
               "&current=temperature_2m,precipitation,weathercode,windspeed_10m"
               "&temperature_unit=fahrenheit&windspeed_unit=mph&timezone=America/New_York")
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read())
        c = data["current"]
        code = c["weathercode"]
        temp = c["temperature_2m"]
        wind = c["windspeed_10m"]
        precip = c["precipitation"]
        if code == 0:                      label,icon,risk_mod = "Clear",        "☀️", -5
        elif code in [1,2,3]:             label,icon,risk_mod = "Partly Cloudy", "⛅",  0
        elif code in [45,48]:             label,icon,risk_mod = "Foggy",         "🌫️",+18
        elif code in [51,53,55,61,63,65]: label,icon,risk_mod = "Rain",          "🌧️",+15
        elif code in [71,73,75,77]:       label,icon,risk_mod = "Snow",          "❄️", +22
        elif code in [80,81,82]:          label,icon,risk_mod = "Rain Showers",  "🌦️",+12
        elif code in [95,96,99]:          label,icon,risk_mod = "Thunderstorm",  "⛈️",+25
        else:                             label,icon,risk_mod = "Cloudy",        "☁️",  +3
        return {"label":label,"icon":icon,"risk_mod":risk_mod,
                "temp":temp,"wind":wind,"precip":precip,"live":True}
    except:
        return {"label":"Unknown","icon":"🌡️","risk_mod":0,
                "temp":"--","wind":"--","precip":0,"live":False}

weather = get_salem_weather()

# ================================================================
# LOCATION RISK DATA — per Salem neighborhood/intersection
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
        "desc": "Evening commute traffic. Icy conditions in winter significantly raise risk."
    },
}

def adjusted_risk(location, weather_mod, hour):
    """Calculate ML-adjusted risk score for a location given current conditions."""
    base = LOCATION_DATA[location]["base_risk"]
    if 16 <= hour <= 18:          h_mod = +18
    elif 7 <= hour <= 9:          h_mod = +12
    elif hour >= 22 or hour <= 5: h_mod = +8
    elif 10 <= hour <= 14:        h_mod = -5
    else:                         h_mod = 0
    return max(0, min(100, base + h_mod + weather_mod))

def risk_label(score):
    if score >= 70:  return "High Risk",     "#dc2626"
    if score >= 40:  return "Moderate Risk", "#f59e0b"
    return "Low Risk", "#16a34a"

# ================================================================
# SIDEBAR — CLIENT PROFILE
# ================================================================
with st.sidebar:
    st.markdown("### 👤 Client Profile")
    client_location = st.selectbox(
        "📍 Client's Home/Commute Location",
        list(LOCATION_DATA.keys()),
        help="Select the intersection nearest to where the client lives or drives most"
    )
    client_hour = st.slider("🕐 Primary Commute Hour", 0, 23, 8)
    am_pm = "AM" if client_hour < 12 else "PM"
    h12   = client_hour % 12
    h12   = 12 if h12 == 0 else h12
    st.caption(f"Commute time: **{h12}:00 {am_pm}**")

    client_type = st.selectbox("🚗 Policy Type", [
        "Personal Auto", "Commercial Vehicle", "Fleet Policy"
    ])
    years_driving = st.slider("📅 Years Driving in Salem", 0, 20, 3)
    st.divider()
    st.markdown("### 🌤️ Current Conditions")
    weather_src = "Live" if weather["live"] else "Cached"
    st.markdown(f"""
    <div style="background:#f5f3ff;border-radius:10px;padding:12px;font-size:.85rem">
      <b>{weather["icon"]} {weather["label"]}</b>
      <span style="color:#9ca3af;font-size:.72rem;margin-left:6px">{weather_src}</span><br>
      {weather["temp"]}°F · Wind {weather["wind"]} mph · Precip {weather["precip"]} mm<br>
      <span style="color:{'#dc2626' if weather['risk_mod']>=15 else '#d97706' if weather['risk_mod']>=8 else '#16a34a'};font-weight:600">
        {"⚠️ High road risk today" if weather["risk_mod"]>=15 else "⚠️ Elevated risk today" if weather["risk_mod"]>=8 else "✅ Normal conditions"}
      </span>
    </div>""", unsafe_allow_html=True)

# ================================================================
# COMPUTE CLIENT RISK PROFILE
# ================================================================
loc_data    = LOCATION_DATA[client_location]
curr_score  = adjusted_risk(client_location, weather["risk_mod"], client_hour)
base_score  = loc_data["base_risk"]
curr_lbl, curr_col = risk_label(curr_score)
experience_mod = max(0, (10 - min(years_driving, 10)) * 0.5)

# ================================================================
# PREMIUM CALCULATION — realistic, transparent, step-by-step
# ================================================================
# 1. Base premium (Massachusetts avg personal auto)
base_premium = 1_200

# 2. Location risk modifier — from crash data at that intersection
loc_mod_pct = int(loc_data["premium_mod"].replace("%","").replace("+",""))
loc_adj = int(base_premium * loc_mod_pct / 100)

# 3. Severity surcharge — avg severity above 1.0 adds cost
#    Each 0.1 above 1.0 = +$15 surcharge
severity_surcharge = int(max(0, (loc_data["avg_severity"] - 1.0) * 150))

# 4. Commute hour surcharge — peak hour commuters pay more
if 7 <= client_hour <= 9 or 16 <= client_hour <= 18:
    commute_surcharge = 80   # rush hour commuter
elif client_hour >= 22 or client_hour <= 5:
    commute_surcharge = 60   # night driver
else:
    commute_surcharge = 0

# 5. Policy type modifier
policy_mods = {"Personal Auto": 0, "Commercial Vehicle": 350, "Fleet Policy": 600}
policy_adj = policy_mods[client_type]

# 6. Experience discount — more years = lower premium
experience_discount = min(years_driving * 25, 250)

# 7. Total — all integers, no floats
est_premium = base_premium + loc_adj + severity_surcharge + commute_surcharge + policy_adj - experience_discount
est_premium = max(800, est_premium)  # floor at $800

# ================================================================
# CLIENT RISK SUMMARY
# ================================================================
st.markdown(f"### Risk Profile: **{client_location}**")
st.markdown(f"*{loc_data['zone']} zone · {client_type} · {years_driving} years driving*")
st.markdown("<br>", unsafe_allow_html=True)

# Live weather risk banner
w_bg  = "#fee2e2" if weather["risk_mod"] >= 15 else "#fef3c7" if weather["risk_mod"] >= 8 else "#f0fdf4"
w_bdr = "#dc2626" if weather["risk_mod"] >= 15 else "#d97706" if weather["risk_mod"] >= 8 else "#16a34a"
st.markdown(f"""
<div style="background:{w_bg};border:1px solid {w_bdr};border-radius:10px;
            padding:12px 18px;margin-bottom:16px;font-size:.85rem">
  <b>{weather["icon"]} Current Salem Weather:</b> {weather["label"]} · {weather["temp"]}°F ·
  Wind {weather["wind"]} mph
  {"— <b>⚠️ Wet/icy conditions increase crash risk by +" + str(weather["risk_mod"]) + " points today</b>" if weather["risk_mod"] > 0
   else " — ✅ Good driving conditions"}
  <span style="color:#9ca3af;font-size:.72rem;margin-left:8px">({'Live data' if weather['live'] else 'Cached'})</span>
</div>""", unsafe_allow_html=True)

# Risk score cards
r1,r2,r3,r4 = st.columns(4)

r1.markdown(f"""
<div style="background:white;border:2px solid {curr_col};border-radius:14px;padding:18px;text-align:center">
  <div style="font-size:.75rem;color:#9ca3af;margin-bottom:4px">CURRENT RISK SCORE</div>
  <div style="font-family:'Syne',sans-serif;font-size:2.5rem;font-weight:800;color:{curr_col}">{curr_score}</div>
  <div style="font-size:.8rem;font-weight:600;color:{curr_col}">{curr_lbl}</div>
  <div style="font-size:.7rem;color:#9ca3af;margin-top:4px">incl. live weather + commute hour</div>
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
| **Base Premium** | $1,200 | Massachusetts average personal auto policy |
| **Location Adjustment ({loc_data['premium_mod']})** | ${loc_adj:+,} | {client_location} has {loc_data['crashes']} recorded crashes |
| **Severity Surcharge** | +${severity_surcharge} | Avg crash severity {loc_data['avg_severity']}/3.0 at this location |
| **Commute Hour** | +${commute_surcharge} | {"Rush hour commuter (7-9AM / 4-6PM)" if commute_surcharge == 80 else "Night driver (10PM-5AM)" if commute_surcharge == 60 else "Off-peak commuter — no surcharge"} |
| **Policy Type** | +${policy_adj} | {client_type} |
| **Experience Discount** | -${min(years_driving*25,250):,} | {years_driving} years driving in Salem (max -$250) |
| **Estimated Annual Premium** | **${est_premium:,}** | Rounded, minimum $800 |

> *Note: This is an ML-informed estimate based on Salem crash data patterns.
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
c1,c2 = st.columns(2)

with c1:
    st.markdown(f"**Risk Score by Hour of Day — {client_location.split('/')[0].strip()}**")
    hours = list(range(24))
    hourly_scores = [adjusted_risk(client_location, weather["risk_mod"], h) for h in hours]
    bar_colors = ["#dc2626" if s>=70 else "#f59e0b" if s>=40 else "#22c55e" for s in hourly_scores]
    fig_h = go.Figure(go.Bar(x=hours, y=hourly_scores, marker_color=bar_colors))
    # Mark client's commute hour
    fig_h.add_vline(x=client_hour, line_dash="dash", line_color="#7c3aed",
                    annotation_text=f"Client commute ({h12}:00 {am_pm})",
                    annotation_position="top right")
    fig_h.add_hline(y=70, line_dash="dot", line_color="#dc2626",
                    annotation_text="High Risk threshold")
    fig_h.add_hline(y=40, line_dash="dot", line_color="#f59e0b",
                    annotation_text="Moderate threshold")
    fig_h.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Hour of Day", showgrid=False, tickvals=list(range(0,24,2))),
        yaxis=dict(title="Risk Score", range=[0,105], gridcolor="#f3f4f6"),
        showlegend=False)
    st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar":False})

with c2:
    st.markdown("**Crash Cause Breakdown — This Location**")
    wet_pct   = loc_data["wet_crashes_pct"]
    night_pct = loc_data["night_crashes_pct"]
    other_pct = 100 - wet_pct - night_pct
    fig_pie = go.Figure(go.Pie(
        labels=["Wet / Rainy Roads","Night / Low Visibility","Other Conditions"],
        values=[wet_pct, night_pct, other_pct],
        hole=0.55,
        marker_colors=["#2563eb","#7c3aed","#9ca3af"],
        textinfo="label+percent"
    ))
    fig_pie.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h",y=-0.2,font=dict(size=10)))
    st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar":False})

# ================================================================
# COMPARE ALL LOCATIONS
# ================================================================
st.markdown("### 📍 Location Risk Comparison — All Salem Zones")
st.markdown('<div style="font-size:.82rem;color:#6b7280;margin-bottom:12px">Compare your client\'s location against all tracked intersections — adjusted for current weather conditions</div>',
            unsafe_allow_html=True)

comp_data = []
for loc, d in LOCATION_DATA.items():
    score = adjusted_risk(loc, weather["risk_mod"], client_hour)
    lbl, col = risk_label(score)
    comp_data.append({
        "Location": loc, "Zone": d["zone"],
        "Current Risk Score": score,
        "Base Risk": d["base_risk"],
        "Risk Level": lbl,
        "Crashes": d["crashes"],
        "Peak Hour": d["peak_hour"],
        "Premium Adjustment": d["premium_mod"],
        "Selected": "★ YOUR CLIENT" if loc == client_location else ""
    })

comp_df = pd.DataFrame(comp_data).sort_values("Current Risk Score", ascending=False)

# Bar chart comparison
bar_cols = ["#7c3aed" if r == client_location else
            "#dc2626" if s >= 70 else "#f59e0b" if s >= 40 else "#22c55e"
            for r, s in zip(comp_df["Location"], comp_df["Current Risk Score"])]

fig_comp = go.Figure(go.Bar(
    x=[l.split("/")[0].strip() + "..." if len(l) > 20 else l for l in comp_df["Location"]],
    y=comp_df["Current Risk Score"],
    marker_color=bar_cols,
    text=comp_df["Current Risk Score"],
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
st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar":False})

# Table
def style_risk_lbl(val):
    if "High" in val:     return "color:#dc2626;font-weight:bold"
    if "Moderate" in val: return "color:#d97706;font-weight:bold"
    return "color:#16a34a;font-weight:bold"

def style_selected(val):
    if val == "★ YOUR CLIENT": return "background:#f5f3ff;color:#7c3aed;font-weight:bold"
    return ""

styled_comp = comp_df[["Location","Zone","Current Risk Score","Risk Level",
                        "Crashes","Peak Hour","Premium Adjustment","Selected"]].style \
    .applymap(style_risk_lbl, subset=["Risk Level"]) \
    .applymap(style_selected, subset=["Selected"]) \
    .background_gradient(subset=["Current Risk Score"], cmap="RdYlGn_r", vmin=0, vmax=100) \
    .hide(axis="index")
st.dataframe(styled_comp, use_container_width=True, height=380)

# ================================================================
# MODEL PERFORMANCE SUMMARY
# ================================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🤖 ML Model Performance Summary")
st.markdown('<div style="font-size:.85rem;color:#6b7280;margin-bottom:12px">Actual model results — Salem crash data with features: Hour, Month, Is_Weekend, Is_Rush_Hour, At_Intersection</div>',
            unsafe_allow_html=True)

models     = ["Dummy (Baseline)","Logistic Regression","Logistic Regression + SMOTE",
              "Random Forest (base)","Random Forest + SMOTE",
              "Binary RF (balanced CV)","RF Cross-Validation (5-fold avg)","Tuned RF (GridSearch)"]
accuracies = [74.3,33.0,39.0,56.0,52.0,55.0,67.0,71.0]
bar_colors_m = ["#9ca3af" if m=="Dummy (Baseline)" else
                "#7c3aed" if m=="Tuned RF (GridSearch)" else "#2563eb"
                for m in models]

fig_m = go.Figure(go.Bar(y=models,x=accuracies,orientation="h",
    marker_color=bar_colors_m,text=[f"{a}%" for a in accuracies],textposition="outside"))
fig_m.add_vline(x=74.3,line_dash="dash",line_color="#9ca3af",
                annotation_text="Dummy baseline (misleading)",annotation_position="top right")
fig_m.update_layout(height=360,margin=dict(l=0,r=70,t=20,b=0),
    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(range=[0,88],ticksuffix="%",gridcolor="#f3f4f6"),
    yaxis=dict(showgrid=False),showlegend=False)
st.plotly_chart(fig_m,use_container_width=True,config={"displayModeBar":False})

st.info("The Dummy classifier's 74.3% is misleading — it only predicts the majority class. "
        "The best-performing model is the Tuned Random Forest at 71%.", icon="ℹ️")

st.markdown("""<div class="footer">
  🛡️ <strong>MataVision</strong> · Insurance Analytics · Salem, MA Capstone Project
</div>""", unsafe_allow_html=True)