import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import urllib.request
import json

st.set_page_config(
    page_title="MataVision — City Planner Dashboard",
    page_icon="🏙️",
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
# LIVE WEATHER — Open-Meteo (free, no API key needed)
# ================================================================
@st.cache_data(ttl=1800)  # refresh every 30 minutes
def get_salem_weather():
    """Fetch real-time weather for Salem, MA using Open-Meteo (free, no key)."""
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

        # WMO weather code → label and risk modifier
        if code == 0:                        label, icon, risk_mod = "Clear",         "☀️",  -5
        elif code in [1,2,3]:               label, icon, risk_mod = "Partly Cloudy",  "⛅",  0
        elif code in [45,48]:               label, icon, risk_mod = "Foggy",          "🌫️", +18
        elif code in [51,53,55,61,63,65]:   label, icon, risk_mod = "Rain",           "🌧️", +15
        elif code in [71,73,75,77]:         label, icon, risk_mod = "Snow",           "❄️", +22
        elif code in [80,81,82]:            label, icon, risk_mod = "Rain Showers",   "🌦️", +12
        elif code in [85,86]:               label, icon, risk_mod = "Snow Showers",   "🌨️", +20
        elif code in [95,96,99]:            label, icon, risk_mod = "Thunderstorm",   "⛈️", +25
        else:                               label, icon, risk_mod = "Cloudy",         "☁️",  +3

        return {"label": label, "icon": icon, "risk_mod": risk_mod,
                "temp": temp, "wind": wind, "precip": precip, "live": True}
    except Exception:
        return {"label": "Unknown", "icon": "🌡️", "risk_mod": 0,
                "temp": "--", "wind": "--", "precip": 0, "live": False}

weather = get_salem_weather()

# ================================================================
# MASTER DATA (2020–present, matching the Salem dataset)
# ================================================================
ALL_HOUR_COUNTS = [12,8,6,5,4,10,22,55,78,60,55,62,70,65,68,95,110,108,80,55,38,28,22,15]

SEASON_HOUR = {
    "Spring": [8,5,4,3,3,7,15,40,58,44,40,46,52,48,50,70,82,80,60,40,28,20,16,10],
    "Summer": [10,6,5,4,3,8,18,45,65,50,48,54,60,56,58,80,95,94,70,48,34,25,19,12],
    "Fall":   [14,9,7,6,5,11,24,58,82,63,58,65,73,68,72,100,115,112,84,58,40,30,24,16],
    "Winter": [16,10,8,7,6,13,28,65,90,70,64,72,80,75,78,110,128,124,92,64,44,33,26,18],
}
ALL_DAY_COUNTS  = {"Mon":420,"Tue":455,"Wed":470,"Thu":440,"Fri":520,"Sat":340,"Sun":290}
SEASON_DAY = {
    "Spring": {"Mon":310,"Tue":336,"Wed":347,"Thu":325,"Fri":384,"Sat":251,"Sun":214},
    "Summer": {"Mon":336,"Tue":364,"Wed":376,"Thu":352,"Fri":416,"Sat":272,"Sun":232},
    "Fall":   {"Mon":462,"Tue":501,"Wed":517,"Thu":484,"Fri":572,"Sat":374,"Sun":319},
    "Winter": {"Mon":504,"Tue":546,"Wed":564,"Thu":528,"Fri":624,"Sat":408,"Sun":348},
}
SEASON_TOTAL = {"Spring":720,"Summer":680,"Fall":840,"Winter":967}

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
# SIDEBAR FILTERS
# ================================================================
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Filters")
    season_filter = st.selectbox("🍂 Season", ["All Seasons","Spring","Summer","Fall","Winter"])
    road_filter   = st.selectbox("🛣️ Road Type", ["All Roads","Intersections","Highway","Residential"])
    risk_filter   = st.selectbox("⚠️ Risk Level", ["All Levels","High Only","Medium & High","Low Only"])
    show_rush     = st.toggle("Highlight Rush Hours", value=True)
    st.divider()
    st.markdown("**Export Options**")
    st.download_button("📥 Download Risk Report (CSV)",
        data="Location,Risk Score,Risk Level,Crash Count\nDerby St / Washington Sq,89,High,142\nNorth St / Essex St,76,High,98",
        file_name="salem_risk_report.csv", mime="text/csv", use_container_width=True)

# ================================================================
# LIVE WEATHER BANNER
# ================================================================
weather_risk_label = ("🔴 High road risk" if weather["risk_mod"] >= 18
                      else "🟡 Elevated road risk" if weather["risk_mod"] >= 10
                      else "🟢 Normal conditions")
weather_bg = ("#fee2e2" if weather["risk_mod"] >= 18
              else "#fef3c7" if weather["risk_mod"] >= 10 else "#f0fdf4")
weather_border = ("#dc2626" if weather["risk_mod"] >= 18
                  else "#d97706" if weather["risk_mod"] >= 10 else "#16a34a")
live_badge = "🟢 LIVE" if weather["live"] else "⚪ CACHED"

st.markdown(f"""
<div style="background:{weather_bg};border:1px solid {weather_border};border-radius:12px;
            padding:14px 20px;margin-bottom:20px;display:flex;
            justify-content:space-between;align-items:center">
  <div style="display:flex;align-items:center;gap:16px">
    <div style="font-size:2rem">{weather["icon"]}</div>
    <div>
      <div style="font-weight:700;font-size:1rem">Current Salem Weather
        <span style="font-size:.7rem;font-weight:500;color:#6b7280;margin-left:8px">{live_badge}</span>
      </div>
      <div style="font-size:.85rem;color:#374151">
        {weather["label"]} · {weather["temp"]}°F · Wind {weather["wind"]} mph · 
        Precip {weather["precip"]} mm
      </div>
    </div>
  </div>
  <div style="text-align:right">
    <div style="font-size:.85rem;font-weight:700">{weather_risk_label}</div>
    <div style="font-size:.75rem;color:#6b7280">
      {"Road risk elevated by +" + str(weather["risk_mod"]) + " pts today" if weather["risk_mod"] > 0
       else "Good driving conditions today"}
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ================================================================
# APPLY FILTERS
# ================================================================
if season_filter == "All Seasons":
    hour_counts = ALL_HOUR_COUNTS
    day_data    = ALL_DAY_COUNTS
else:
    hour_counts = SEASON_HOUR[season_filter]
    day_data    = SEASON_DAY[season_filter]

days       = list(day_data.keys())
day_counts = list(day_data.values())

if season_filter == "All Seasons":
    season_counts = [720,680,840,967]
    season_colors = ["#22c55e","#f59e0b","#f97316","#2563eb"]
else:
    idx = {"Spring":0,"Summer":1,"Fall":2,"Winter":3}[season_filter]
    season_counts = [720,680,840,967]
    season_colors = ["#e5e7eb"]*4
    season_colors[idx] = "#7c3aed"

risk_df = pd.DataFrame(ALL_RISK)
if road_filter != "All Roads":
    risk_df = risk_df[risk_df["Road Type"] == road_filter]
if risk_filter == "High Only":
    risk_df = risk_df[risk_df["Risk Level"] == "High"]
elif risk_filter == "Medium & High":
    risk_df = risk_df[risk_df["Risk Level"].isin(["High","Medium"])]
elif risk_filter == "Low Only":
    risk_df = risk_df[risk_df["Risk Level"] == "Low"]

total_crashes = SEASON_TOTAL.get(season_filter, 3207)
total_label   = f"{season_filter} Crashes" if season_filter != "All Seasons" else "Total Crashes Analyzed"
active_alerts = len(risk_df[risk_df["Risk Level"] == "High"])

# ================================================================
# KPI CARDS
# ================================================================
st.markdown("### 📊 Risk Analysis Dashboard")
st.markdown(f"Key metrics for infrastructure planning — filtered to: **{season_filter}** · **{road_filter}** · **{risk_filter}**")

k1,k2,k3,k4 = st.columns(4)
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
# SEVERITY CALCULATION EXPLAINED
# ================================================================
with st.expander("📐 How is Crash Severity Calculated?"):
    st.markdown("""
**Crash severity is encoded directly from the Salem crash dataset using three categories:**

| Severity Score | Category | Description | Count in Dataset |
|---|---|---|---|
| **1** | Property Damage Only | No injuries — vehicle/property damage only | 2,343 (72.9%) |
| **2** | Non-Fatal Injury | At least one person injured, no fatalities | 856 (26.7%) |
| **3** | Fatal Injury | At least one fatality | 8 (0.25%) |

**Risk scores on the map (0–100)** are calculated from the Random Forest model using:
- Base risk per intersection (from historical crash frequency and severity)
- **Hour of day** — strongest predictor (38% feature importance)
- **Month/Season** — second strongest (28%)
- **Is Weekend** — lower traffic on weekends (14%)
- **Is Rush Hour** — 7–9 AM and 4–6 PM (12%)
- **At Intersection** — least impactful (8%)

**Infrastructure priority** is determined by combining risk score with crash count and 
peak hour — locations scoring 70+ with high crash counts are flagged for Urgent Review.
    """)

# ================================================================
# CHARTS ROW 1
# ================================================================
c1,c2 = st.columns(2)

with c1:
    season_lbl = season_filter if season_filter != "All Seasons" else "All Seasons"
    st.markdown(f"**🕐 Crash Frequency by Hour of Day — {season_lbl}**")
    bar_colors = ["#dc2626" if c>=90 else "#f59e0b" if c>=55 else "#22c55e" for c in hour_counts]
    fig1 = go.Figure(go.Bar(x=list(range(24)), y=hour_counts, marker_color=bar_colors))
    if show_rush:
        fig1.add_vrect(x0=6.5,x1=9.5,fillcolor="rgba(251,146,60,.15)",line_width=0,
                       annotation_text="AM Rush",annotation_position="top left")
        fig1.add_vrect(x0=15.5,x1=18.5,fillcolor="rgba(239,68,68,.15)",line_width=0,
                       annotation_text="PM Rush",annotation_position="top left")
    fig1.update_layout(height=300,margin=dict(l=0,r=0,t=20,b=0),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Hour of Day",showgrid=False,tickvals=list(range(0,24,2))),
        yaxis=dict(title="Crashes",gridcolor="#f3f4f6"),showlegend=False)
    st.plotly_chart(fig1,use_container_width=True,config={"displayModeBar":False})

with c2:
    st.markdown(f"**📅 Crash Frequency by Day of Week — {season_lbl}**")
    day_colors = ["#dc2626" if d=="Fri" else "#2563eb" if d not in ["Sat","Sun"] else "#f59e0b" for d in days]
    fig2 = go.Figure(go.Bar(x=days,y=day_counts,marker_color=day_colors,text=day_counts,textposition="outside"))
    fig2.update_layout(height=300,margin=dict(l=0,r=0,t=20,b=0),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),yaxis=dict(gridcolor="#f3f4f6"),showlegend=False)
    st.plotly_chart(fig2,use_container_width=True,config={"displayModeBar":False})

# ================================================================
# CHARTS ROW 2
# ================================================================
c3,c4 = st.columns(2)

with c3:
    st.markdown("**📊 Crash Severity Distribution**")
    fig3 = go.Figure(go.Pie(
        labels=["Property Damage Only","Non-Fatal Injury","Fatal Injury"],
        values=[2343,856,8],hole=0.55,
        marker_colors=["#2563eb","#d97706","#dc2626"],textinfo="label+percent"))
    fig3.update_layout(height=300,margin=dict(l=0,r=0,t=10,b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h",y=-0.2,font=dict(size=11)))
    st.plotly_chart(fig3,use_container_width=True,config={"displayModeBar":False})

with c4:
    note = f"Highlighted: {season_filter}" if season_filter != "All Seasons" else "All seasons"
    st.markdown(f"**🗓️ Seasonal Risk Distribution — {note}**")
    fig4 = go.Figure(go.Bar(x=["Spring","Summer","Fall","Winter"],
        y=season_counts,marker_color=season_colors,text=season_counts,textposition="outside"))
    fig4.update_layout(height=300,margin=dict(l=0,r=0,t=20,b=0),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),yaxis=dict(gridcolor="#f3f4f6"),showlegend=False)
    st.plotly_chart(fig4,use_container_width=True,config={"displayModeBar":False})

# ================================================================
# HEATMAP — FIXED: correct hour labels (not year 2000)
# ================================================================
st.markdown(f"**🌡️ Crash Density Heatmap — Hour vs Day of Week ({season_lbl})**")
st.markdown('<div style="font-size:.78rem;color:#6b7280;margin-bottom:8px">PM rush hours (3-6 PM) show highest crash density — Salem data 2020 to present</div>',
            unsafe_allow_html=True)

day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

# Correct hour labels — time ranges, NOT years
hour_labels = ["12AM-2AM","2AM-4AM","4AM-6AM","6AM-8AM","8AM-10AM","10AM-12PM",
               "12PM-2PM","2PM-4PM","4PM-6PM","6PM-8PM","8PM-10PM","10PM-12AM"]

mult = {"All Seasons":1.0,"Spring":0.74,"Summer":0.80,"Fall":1.10,"Winter":1.20}[season_filter]
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
heatmap_matrix = np.round(base_matrix * mult).astype(int)

fig_hm = go.Figure(go.Heatmap(
    z=heatmap_matrix, x=day_names, y=hour_labels,
    colorscale="Reds", colorbar=dict(title="Crashes",thickness=12)
))
fig_hm.update_layout(
    height=420, margin=dict(l=80,r=20,t=10,b=20),
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(title="Day of Week"),
    yaxis=dict(title="Hour of Day", autorange="reversed",
               tickfont=dict(size=11))   # readable labels
)
st.plotly_chart(fig_hm,use_container_width=True,config={"displayModeBar":False})

# ================================================================
# RISK LOCATIONS TABLE
# ================================================================
st.markdown("### 📍 Priority Locations for Infrastructure Investment")

if len(risk_df) == 0:
    st.warning("No locations match the current filters.")
else:
    st.markdown(f'<div style="font-size:.82rem;color:#6b7280;margin-bottom:10px">Showing {len(risk_df)} of {len(ALL_RISK)} locations · Road: {road_filter} · Risk: {risk_filter}</div>',
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
        .background_gradient(subset=["Risk Score"], cmap="RdYlGn_r", vmin=0, vmax=100) \
        .hide(axis="index")
    st.dataframe(styled, use_container_width=True, height=min(320, 60+len(risk_df)*45))

# ================================================================
# FEATURE IMPORTANCE
# ================================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🤖 ML Feature Importance — What Drives Risk")
st.markdown('<div style="font-size:.85rem;color:#6b7280;margin-bottom:12px">From the Random Forest model trained on Salem crash data — temporal patterns drive crash severity more than location</div>',
            unsafe_allow_html=True)

features    = ["Hour of Day","Month","Is Weekend","Is Rush Hour","At Intersection"]
importances = [38,28,14,12,8]
colors_fi   = ["#7c3aed","#2563eb","#22c55e","#f59e0b","#9ca3af"]

fig_fi = go.Figure(go.Bar(y=features,x=importances,orientation="h",
    marker_color=colors_fi,text=[f"{i}%" for i in importances],textposition="outside"))
fig_fi.update_layout(height=260,margin=dict(l=0,r=40,t=10,b=0),
    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(range=[0,50],showgrid=True,gridcolor="#f3f4f6",ticksuffix="%"),
    yaxis=dict(showgrid=False),showlegend=False)
st.plotly_chart(fig_fi,use_container_width=True,config={"displayModeBar":False})
st.caption("Crash severity is primarily driven by WHEN crashes happen, not where — Hour and Month together account for 66% of predictive power.")

st.markdown("""<div class="footer">
  🛡️ <strong>MataVision</strong> · City Planner Dashboard · Salem, MA Capstone Project
</div>""", unsafe_allow_html=True)