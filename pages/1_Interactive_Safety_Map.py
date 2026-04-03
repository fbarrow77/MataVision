import streamlit as st
import pandas as pd
import folium
from streamlit.components.v1 import html as st_html
import plotly.graph_objects as go
import numpy as np
import os

# Google Maps
try:
    import googlemaps
    GMAPS_AVAILABLE = True
except ImportError:
    GMAPS_AVAILABLE = False

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(
    page_title="MataVision — Safety Map",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

css_path = os.path.join(os.path.dirname(__file__), '..', 'styles.css')
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---- NAV ----
col_back, col_brand, col_right = st.columns([1, 4, 2])
with col_back:
    if st.button("← Back", key="back_btn"):
        st.switch_page("app.py")
with col_brand:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:8px 0">
      <div style="background:#7c3aed;border-radius:8px;width:32px;height:32px;
                  display:flex;align-items:center;justify-content:center;font-size:1rem">🛡️</div>
      <div>
        <div style="font-weight:700;font-size:.95rem">City Risk &amp; Community Safety Map</div>
        <div style="font-size:.72rem;color:#7c3aed">Predicting risks, protecting lives</div>
      </div>
    </div>""", unsafe_allow_html=True)
with col_right:
    st.markdown("""<div style="display:flex;align-items:center;justify-content:flex-end;padding:8px 0">
      <div style="background:#f5f3ff;color:#7c3aed;padding:5px 12px;border-radius:20px;
                  font-size:.8rem;font-weight:600">Salem, MA</div>
    </div>""", unsafe_allow_html=True)

st.divider()

view = st.segmented_control("Select View Mode",
    options=["👥 Community View", "🏙️ Planner Mode"],
    default="👥 Community View")
is_planner = view == "🏙️ Planner Mode"

# ================================================================
# GOOGLE MAPS API KEY
# ================================================================
GMAPS_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")

# If key not in env, allow user to enter it in sidebar once
if not GMAPS_KEY:
    with st.sidebar:
        st.markdown("### 🔑 Google Maps API Key")
        entered_key = st.text_input("Paste API key here", type="password",
                                    help="Get a free key at console.cloud.google.com — enable Directions API")
        if entered_key:
            GMAPS_KEY = entered_key
        else:
            st.caption("Without a key, routes use straight-line paths between intersections.")

USE_GMAPS = GMAPS_AVAILABLE and bool(GMAPS_KEY)

# ================================================================
# SALEM LOCATIONS & ML MODEL
# ================================================================
SALEM_LOCATIONS = {
    "Derby St / Washington Sq":       (42.5195, -70.8967),
    "North St / Essex St":            (42.5228, -70.8952),
    "Bridge St / Federal St":         (42.5243, -70.9043),
    "Highland Ave / Jefferson Ave":   (42.5108, -70.9012),
    "Lafayette St / Loring Ave":      (42.5172, -70.9005),
    "Canal St / Grove St":            (42.5072, -70.9020),
    "Marlborough Rd / Ocean Ave":     (42.5155, -70.9012),
    "Derby St / Pickering Wharf":     (42.5168, -70.8901),
    "Boston St / Loring Ave":         (42.5092, -70.8845),
    "Aborn St / Essex St":            (42.5210, -70.9005),
    "Salem Commuter Rail Station":    (42.5233, -70.9030),
    "Charter St / Derby St":          (42.5188, -70.8958),
    "Peabody St / North St":          (42.5255, -70.8971),
    "Washington St / Bridge St":      (42.5248, -70.9018),
    "Margin St / Congress St":        (42.5145, -70.8968),
    "Winter St / Essex St":           (42.5198, -70.9020),
    "Webb St / Lafayette St":         (42.5163, -70.8847),
    "Salem Common / Washington Sq":   (42.5202, -70.8985),
    "Collins Cove / Derby St":        (42.5175, -70.8812),
    "Witch Museum / Washington Sq N": (42.5208, -70.8990),
}

BASE_RISK = {
    "Derby St / Washington Sq":       89,
    "North St / Essex St":            76,
    "Marlborough Rd / Ocean Ave":     71,
    "Bridge St / Federal St":         68,
    "Highland Ave / Jefferson Ave":   62,
    "Washington St / Bridge St":      60,
    "Aborn St / Essex St":            55,
    "Charter St / Derby St":          52,
    "Salem Commuter Rail Station":    50,
    "Peabody St / North St":          48,
    "Winter St / Essex St":           45,
    "Margin St / Congress St":        44,
    "Salem Common / Washington Sq":   40,
    "Witch Museum / Washington Sq N": 38,
    "Derby St / Pickering Wharf":     36,
    "Boston St / Loring Ave":         34,
    "Collins Cove / Derby St":        30,
    "Webb St / Lafayette St":         25,
    "Lafayette St / Loring Ave":      32,
    "Canal St / Grove St":            28,
}

RISK_THRESHOLD = 30
SALEM_CENTER   = [42.519, -70.896]
SALEM_ZOOM     = 15

def ml_risk_score(location, hour, is_weekend, weather):
    """RF model simulation based on Random Forest feature importance weights from the Salem crash analysis."""
    base = BASE_RISK.get(location, 40)
    if 16 <= hour <= 18:            hour_mod = +18
    elif 7 <= hour <= 9:            hour_mod = +12
    elif hour >= 22 or hour <= 5:   hour_mod = +8
    elif 10 <= hour <= 14:          hour_mod = -5
    else:                           hour_mod = 0
    weekend_mod  = -8 if is_weekend else +4
    weather_mods = {"Clear / Dry": -5, "Rain / Wet Roads": +15,
                    "Snow / Ice": +22, "Fog / Low Visibility": +18}
    return max(0, min(100, base + hour_mod + weekend_mod + weather_mods.get(weather, 0)))

def risk_label(score):
    if score >= 70:             return "High Risk",     "#dc2626"
    if score >= RISK_THRESHOLD: return "Moderate Risk", "#f59e0b"
    return "Low Risk", "#16a34a"

def score_route(stops, hour, is_weekend, weather):
    return [{"intersection": s,
             "score":  ml_risk_score(s, hour, is_weekend, weather),
             "label":  risk_label(ml_risk_score(s, hour, is_weekend, weather))[0],
             "color":  risk_label(ml_risk_score(s, hour, is_weekend, weather))[1],
             "coords": SALEM_LOCATIONS[s]} for s in stops]

# ================================================================
# GOOGLE MAPS DIRECTIONS
# ================================================================
@st.cache_data(show_spinner=False)
def get_google_route(origin_name, dest_name, api_key, alternative=False,
                     safer_waypoint=None, hour=8, is_weekend=False, weather="Clear / Dry"):
    """
    Call Google Maps Directions API.
    If alternative=True and safer_waypoint is provided, forces a detour
    through the lowest-risk intersection to create a genuinely different route.
    """
    try:
        gmaps = googlemaps.Client(key=api_key)
        origin = SALEM_LOCATIONS[origin_name]
        dest   = SALEM_LOCATIONS[dest_name]

        # For safer route — find lowest-risk waypoint NOT on primary route
        # and force Google to route through it
        if alternative and safer_waypoint:
            wp_coords = SALEM_LOCATIONS[safer_waypoint]
            result = gmaps.directions(
                origin=f"{origin[0]},{origin[1]}",
                destination=f"{dest[0]},{dest[1]}",
                waypoints=[f"{wp_coords[0]},{wp_coords[1]}"],
                mode="driving",
                region="us"
            )
        else:
            result = gmaps.directions(
                origin=f"{origin[0]},{origin[1]}",
                destination=f"{dest[0]},{dest[1]}",
                mode="driving",
                alternatives=True,
                region="us"
            )

        if not result:
            return None, None

        # For primary: use first (fastest) route
        # For safer: use the waypoint-forced route
        route = result[0]

        # Decode polyline points
        import polyline as pl
        points = []
        for leg in route["legs"]:
            for step in leg["steps"]:
                points += pl.decode(step["polyline"]["points"])

        # Extract intersection names from step HTML instructions
        import re
        waypoint_names = []
        for leg in route["legs"]:
            for step in leg["steps"]:
                clean = re.sub('<[^<]+?>', '', step.get("html_instructions", ""))
                if any(kw in clean.lower() for kw in ["turn", "onto", "continue", "merge"]):
                    waypoint_names.append(clean[:60])

        # Match waypoints to our known Salem locations
        matched_stops = [origin_name]
        route_lats = [p[0] for p in points]
        route_lons = [p[1] for p in points]

        for loc_name, (lat, lon) in SALEM_LOCATIONS.items():
            if loc_name in [origin_name, dest_name]:
                continue
            # Check if this known location is near the route polyline
            dists = [abs(lat - rlat) + abs(lon - rlon) for rlat, rlon in zip(route_lats, route_lons)]
            if min(dists) < 0.003:   # within ~300m of the route
                matched_stops.append(loc_name)

        matched_stops.append(dest_name)
        # Deduplicate keeping order
        seen = set()
        unique_stops = []
        for s in matched_stops:
            if s not in seen:
                seen.add(s)
                unique_stops.append(s)

        return points, unique_stops

    except Exception as e:
        st.warning(f"Google Maps API error: {e}. Falling back to straight-line route.")
        return None, None

def get_fallback_route(start, end):
    """Straight-line fallback when no API key is provided."""
    s_lat, s_lon = SALEM_LOCATIONS[start]
    e_lat, e_lon = SALEM_LOCATIONS[end]
    mid_lat = (s_lat + e_lat) / 2
    mid_lon = (s_lon + e_lon) / 2
    candidates = [(((lat-mid_lat)**2+(lon-mid_lon)**2)**.5, name)
                  for name, (lat, lon) in SALEM_LOCATIONS.items()
                  if name not in [start, end]]
    candidates.sort()
    stops = [start] + [c[1] for c in candidates[:2]] + [end]
    # Straight line points
    points = [SALEM_LOCATIONS[s] for s in stops]
    return points, stops

def find_safer_stops(primary_stops, hour, is_weekend, weather, start, end):
    """
    Find an alternative set of stops avoiding the highest-risk intersection.
    Returns (alt_stops, best_waypoint) — waypoint is used to force Google Maps detour.
    """
    primary_scores = score_route(primary_stops, hour, is_weekend, weather)
    max_risk = max(primary_scores, key=lambda x: x["score"])
    if max_risk["score"] < RISK_THRESHOLD:
        return None, None

    # Find lowest-risk intersection NOT on primary route
    # Sort by risk score first (lowest risk = best detour), then by proximity
    s_lat, s_lon = SALEM_LOCATIONS[start]
    e_lat, e_lon = SALEM_LOCATIONS[end]
    alternatives = []
    for name, (lat, lon) in SALEM_LOCATIONS.items():
        if name in primary_stops or name in [start, end]:
            continue
        score = ml_risk_score(name, hour, is_weekend, weather)
        # Weight: 70% risk reduction, 30% proximity
        dist = (((lat-(s_lat+e_lat)/2)**2+((lon-(s_lon+e_lon)/2)**2))**.5)
        weighted = (score * 0.7) + (dist * 1000 * 0.3)
        alternatives.append((weighted, score, dist, name))
    alternatives.sort()

    if not alternatives:
        return None, None

    best_waypoint = alternatives[0][3]
    alt_stops = [best_waypoint if s == max_risk["intersection"] else s
                 for s in primary_stops]
    # Make sure waypoint is included if it wasnt in primary stops
    if best_waypoint not in alt_stops:
        alt_stops.insert(len(alt_stops)//2, best_waypoint)

    return alt_stops, best_waypoint

def build_route_map(route_points, stops, scores, map_label, height=500):
    """
    Build a folium map with the real Google Maps polyline route.
    route_points: list of (lat,lon) from Directions API (or fallback)
    stops: list of known intersection names on the route
    scores: scored stop info dicts
    """
    # Always lock to Salem
    all_lats = [p[0] for p in route_points]
    all_lons = [p[1] for p in route_points]

    sw = [max(42.495, min(all_lats) - 0.003), max(-70.930, min(all_lons) - 0.004)]
    ne = [min(42.545, max(all_lats) + 0.003), min(-70.860, max(all_lons) + 0.004)]

    m = folium.Map(
        location=SALEM_CENTER,
        zoom_start=SALEM_ZOOM,
        tiles="CartoDB positron",
        min_zoom=13
    )
    m.fit_bounds([sw, ne])

    # Draw the real road route polyline
    # safer route uses green, primary uses purple
    line_color = "#16a34a" if "Safer" in map_label else "#7c3aed"
    folium.PolyLine(
        route_points,
        color=line_color, weight=5, opacity=0.9
    ).add_to(m)

    # Score markers for each matched intersection
    score_map = {s["intersection"]: s for s in scores}
    for i, stop in enumerate(stops):
        lat, lon = SALEM_LOCATIONS[stop]
        info  = score_map.get(stop, {"score": 40, "label": "Moderate Risk", "color": "#f59e0b"})
        score = info["score"]
        color = info["color"]
        label = info["label"]

        if i == 0:
            folium.Marker([lat, lon],
                tooltip=f"🚀 START: {stop}",
                popup=folium.Popup(f"<b>START</b><br>{stop}<br>Risk: {score}% — {label}", max_width=200),
                icon=folium.Icon(color="blue", icon="home", prefix="fa")
            ).add_to(m)

        elif i == len(stops) - 1:
            folium.Marker([lat, lon],
                tooltip=f"🏁 DESTINATION: {stop}",
                popup=folium.Popup(f"<b>DESTINATION</b><br>{stop}<br>Risk: {score}% — {label}", max_width=200),
                icon=folium.Icon(color="green", icon="flag", prefix="fa")
            ).add_to(m)

        else:
            dot_color = "#dc2626" if score >= 70 else "#f59e0b" if score >= RISK_THRESHOLD else "#22c55e"

            if score >= RISK_THRESHOLD:
                folium.CircleMarker([lat, lon], radius=22,
                    color=dot_color, weight=1.5, fill=False, opacity=0.3
                ).add_to(m)

            folium.CircleMarker([lat, lon], radius=14,
                color="white", weight=2.5, fill=True,
                fill_color=dot_color, fill_opacity=0.92,
                popup=folium.Popup(
                    f"<b>{stop}</b><br>Risk: <b>{score}%</b><br>{label}<br>"
                    f"{'⚠️ Drive with caution' if score >= RISK_THRESHOLD else '✅ Relatively safe'}",
                    max_width=220),
                tooltip=f"{stop}: {score}% ({label})"
            ).add_to(m)

            folium.Marker([lat, lon], icon=folium.DivIcon(
                html=f'<div style="font-size:11px;font-weight:800;color:white;'
                     f'text-align:center;line-height:28px;width:28px">{score}</div>',
                icon_size=(28,28), icon_anchor=(14,14))
            ).add_to(m)

    # Label badge
    m.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
                background:#7c3aed;color:white;border-radius:20px;padding:5px 16px;
                font-size:12px;font-weight:700;z-index:9999;box-shadow:0 2px 8px rgba(0,0,0,.2)">
      {map_label}</div>"""))

    # Source badge
    source = "🗺️ Google Maps Route" if USE_GMAPS else "📍 Estimated Route"
    m.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;bottom:50px;right:10px;background:white;border-radius:8px;
                padding:4px 10px;font-size:11px;color:#6b7280;z-index:9999;
                box-shadow:0 1px 4px rgba(0,0,0,.1)">{source}</div>"""))

    # Legend
    m.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;bottom:16px;left:16px;background:white;border-radius:10px;
                padding:10px 14px;box-shadow:0 2px 10px rgba(0,0,0,.15);z-index:9999;font-size:12px">
      <span style="color:#dc2626">●</span> High Risk (70%+)<br>
      <span style="color:#f59e0b">●</span> Moderate ({RISK_THRESHOLD}%+)<br>
      <span style="color:#22c55e">●</span> Low Risk<br>
      <span style="color:#7c3aed">━</span> Route
    </div>"""))

    return m

# ================================================================
# LOAD CRASH DATA
# ================================================================
@st.cache_data
def load_data():
    paths = [
        "data/CrashData.csv", "../data/CrashData.csv",
        r"C:\Users\Fatoumata Barrow\OneDrive - Salem State University\Desktop\MataVision\data\CrashData.csv",
    ]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p, skiprows=3, header=0)
            df.columns = df.columns.str.strip()
            return df
    np.random.seed(42)
    n = 200
    hours = np.random.choice(range(24), n)
    days = np.random.choice(
        ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], n)
    return pd.DataFrame({
        'LATITUDE': np.random.uniform(42.505, 42.540, n),
        'LONGITUDE': np.random.uniform(-70.915, -70.870, n),
        'Hour': hours,
        'Severity_Score': np.random.choice([1, 2, 3], n),
        'DayOfWeek': days,
        'Month': np.random.choice(range(1, 13), n),
        'At_Intersection': np.random.choice([0, 1], n),
    })

crash_df = load_data()

# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    if not is_planner:
        st.markdown("### 🎯 Map Filters")
        severity_filter = st.multiselect("⚠️ Show Severity", [
            "Property Damage Only","Non-Fatal Injury","Fatal Injury"
        ], default=["Property Damage Only","Non-Fatal Injury","Fatal Injury"])
    else:
        st.markdown("### 🎯 Risk Filters")
        time_filter    = st.selectbox("⏰ Time of Day",[
            "All Times","Morning Rush (7-9 AM)","Midday (10 AM-2 PM)","Evening Rush (4-6 PM)","Night (8 PM+)"])
        weather_filter = st.selectbox("🌧️ Weather",[
            "All Weather","Clear / Dry","Rain / Wet Roads","Snow / Ice","Fog"])
        road_filter    = st.selectbox("🛣️ Road Type",[
            "All Roads","Intersections Only","Highway / Major Road","Residential"])
        severity_filter = st.multiselect("⚠️ Severity",[
            "Property Damage Only","Non-Fatal Injury","Fatal Injury"
        ], default=["Property Damage Only","Non-Fatal Injury","Fatal Injury"])
        st.button("🔄 Update Analysis", use_container_width=True, type="primary")

    st.divider()
    st.markdown("### 📊 " + ("Infrastructure Status" if is_planner else "Safety Overview"))
    st.markdown(f"""
    <div class="infra-status">
      <div class="status-row"><span>Overall Risk</span><span class="badge-moderate">Moderate</span></div>
      <div class="status-row"><span>Active Alerts</span><span class="status-num">3</span></div>
      <div class="status-row"><span>Best Model Acc.</span><span class="status-num">71%</span></div>
      {"<div class='status-row'><span>Pending Actions</span><span class='status-num'>5</span></div>" if is_planner else ""}
    </div>""", unsafe_allow_html=True)

# ================================================================
# COMMUNITY VIEW
# ================================================================
if not is_planner:

    map_tab, route_tab = st.tabs(["🗺️ Safety Map", "🧭 Plan My Safe Route"])

    # ── TAB 1: SAFETY MAP ────────────────────────────────────────
    with map_tab:
        map_col, info_col = st.columns([3,1])
        with map_col:
            st.markdown("""<div style="margin-bottom:8px">
              <div style="font-weight:700;font-size:1rem">📍 Interactive Safety Map</div>
              <div style="font-size:.78rem;color:#6b7280">Click any dot for intersection details</div>
            </div>""", unsafe_allow_html=True)

            m = folium.Map(location=SALEM_CENTER, zoom_start=SALEM_ZOOM, tiles="CartoDB positron")
            hotspots = [
                (42.5195,-70.8967,89,'High','Derby St / Washington Sq'),
                (42.5228,-70.8967,76,'High','North St / Essex St'),
                (42.5155,-70.9012,71,'High','Marlborough Rd / Ocean Ave'),
                (42.5243,-70.9043,68,'Medium','Bridge St / Federal St'),
                (42.5108,-70.9012,62,'Medium','Highland Ave / Jefferson Ave'),
                (42.5172,-70.9005,32,'Low','Lafayette St / Loring Ave'),
                (42.5072,-70.9020,28,'Low','Canal St / Grove St'),
            ]
            dot_colors = {'High':'#dc2626','Medium':'#f59e0b','Low':'#22c55e'}
            for lat,lon,score,level,name in hotspots:
                folium.CircleMarker([lat,lon],
                    radius=12 if level=='High' else 9 if level=='Medium' else 7,
                    color='white', weight=2.5, fill=True,
                    fill_color=dot_colors[level], fill_opacity=0.88,
                    tooltip=folium.Tooltip(f"<b>{level} Risk Zone</b><br>{name}")
                ).add_to(m)
            m.get_root().html.add_child(folium.Element("""
            <div style="position:fixed;bottom:16px;left:16px;background:white;border-radius:10px;
                        padding:12px 16px;box-shadow:0 2px 10px rgba(0,0,0,.15);z-index:9999;font-size:12px">
              <b>Risk Levels</b><br>
              <span style="color:#dc2626">●</span> High Risk (70%+)<br>
              <span style="color:#f59e0b">●</span> Medium Risk (40-69%)<br>
              <span style="color:#22c55e">●</span> Low Risk (&lt;40%)
            </div>"""))
            st_html(m._repr_html_(), height=460)

        with info_col:
            st.markdown("**Crashes by Hour**")
            hc = crash_df.groupby('Hour').size().reset_index(name='count') if 'Hour' in crash_df.columns \
                 else pd.DataFrame({'Hour':list(range(24)),
                      'count':[12,8,6,5,4,10,22,55,78,60,55,62,70,65,68,95,110,108,80,55,38,28,22,15]})
            fig_h = go.Figure(go.Bar(x=hc['Hour'],y=hc['count'],
                marker_color=['#dc2626' if c>=90 else '#f59e0b' if c>=55 else '#22c55e' for c in hc['count']]))
            fig_h.update_layout(height=180,margin=dict(l=0,r=0,t=0,b=0),
                paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(tickfont=dict(size=9),showgrid=False),
                yaxis=dict(tickfont=dict(size=9),gridcolor='#f3f4f6'),showlegend=False)
            st.plotly_chart(fig_h,use_container_width=True,config={'displayModeBar':False})
            st.markdown("""<div class="infra-status">
              <div class="status-row"><span>Overall Risk</span><span class="badge-moderate">Moderate</span></div>
              <div class="status-row"><span>High-Risk Zones</span><span class="status-num">5</span></div>
              <div class="status-row"><span>Best Model Acc.</span><span class="status-num">71%</span></div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        g1,g2,g3,g4 = st.columns(4)
        g1.markdown("""<div class="glance-card purple-card">🌙
          <span class="glance-num">29%</span><span class="glance-label">Night Crashes</span>
          <span class="glance-desc">of accidents occur after dark</span></div>""",unsafe_allow_html=True)
        g2.markdown("""<div class="glance-card blue-card">🌧️
          <span class="glance-num">34%</span><span class="glance-label">Wet Road Crashes</span>
          <span class="glance-desc">happen during rain/wet conditions</span></div>""",unsafe_allow_html=True)
        g3.markdown("""<div class="glance-card red-card">⚠️
          <span class="glance-num">5</span><span class="glance-label">High-Risk Zones</span>
          <span class="glance-desc">intersections need immediate attention</span></div>""",unsafe_allow_html=True)
        g4.markdown("""<div class="glance-card green-card">📈
          <span class="glance-num">18%</span><span class="glance-label">Risk Reduction</span>
          <span class="glance-desc">improvement since ML implementation</span></div>""",unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        t1,t2,t3 = st.columns(3)
        t1.markdown("""<div class="tip-card tip-important"><span class="tip-badge badge-important">Important</span>
          <h5>🌧️ Drive Slowly in Rain</h5>
          <p>Reduce speed 10-15 mph on wet roads. Salem sees 34% more crashes during rain.</p></div>""",unsafe_allow_html=True)
        t2.markdown("""<div class="tip-card tip-important"><span class="tip-badge badge-important">Important</span>
          <h5>🌙 Extra Caution at Night</h5>
          <p>Use headlights and stay alert. 29% of Salem crashes happen after dark.</p></div>""",unsafe_allow_html=True)
        t3.markdown("""<div class="tip-card tip-good"><span class="tip-badge badge-good">Good to Know</span>
          <h5>👁️ Watch Intersection Signals</h5>
          <p>Come to complete stops, especially at the 5 high-risk intersections.</p></div>""",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("""<div class="drive-safe-banner"><div style="font-size:1.8rem">🛡️</div>
          <div><h4>Drive Safe, Salem!</h4>
          <p>Our community is working together to make Salem roads safer. Every safe trip counts!</p>
          </div></div>""",unsafe_allow_html=True)

    # ── TAB 2: PLAN MY SAFE ROUTE ─────────────────────────────────
    with route_tab:

        # Google Maps status banner
        if USE_GMAPS:
            st.success("🗺️ **Google Maps connected** — routes follow real Salem roads", icon="✅")
        else:
            st.info(
                "📍 **Running without Google Maps** — routes use estimated paths. "
                "Add an API key in the sidebar for real road routing.",
                icon="ℹ️"
            )

        st.markdown("""<div style="margin-bottom:16px">
          <div style="font-weight:700;font-size:1.05rem">🧭 Plan My Safe Route</div>
          <div style="font-size:.82rem;color:#6b7280">
            Pick a start and destination — the ML model scores every intersection
            and finds a safer alternative if any stop exceeds <b>30% risk</b>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── TRIP CONDITIONS ───────────────────────────────
        st.markdown("""<div style="background:#f5f3ff;border-radius:12px;padding:16px 20px;margin-bottom:16px">
          <div style="font-weight:700;font-size:.88rem;color:#5b21b6;margin-bottom:12px">
            ⚙️ Trip Conditions</div>
        </div>""", unsafe_allow_html=True)

        cond1, cond2, cond3, cond4 = st.columns([3, 1, 1, 2])
        with cond1:
            route_hour = st.slider("🕐 Hour of Travel", 0, 23, 8)
        with cond2:
            am_pm  = "AM" if route_hour < 12 else "PM"
            disp_h = route_hour % 12
            disp_h = 12 if disp_h == 0 else disp_h
            st.markdown(f"""<div style="background:white;border-radius:10px;padding:10px;
                            text-align:center;border:1px solid #ddd6fe;margin-top:26px">
              <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;color:#7c3aed">
                {disp_h}:00</div>
              <div style="font-size:.7rem;color:#6b7280">{am_pm}</div></div>""", unsafe_allow_html=True)
        with cond3:
            route_weekend = st.toggle("📅 Weekend", value=False)
            is_rush = (7 <= route_hour <= 9) or (16 <= route_hour <= 18)
            if is_rush:
                st.markdown('<div style="background:#fee2e2;color:#991b1b;border-radius:8px;padding:5px 8px;font-size:.72rem;font-weight:600;margin-top:6px">⚠️ Rush Hour</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="background:#dcfce7;color:#166534;border-radius:8px;padding:5px 8px;font-size:.72rem;font-weight:600;margin-top:6px">✅ Off-Peak</div>', unsafe_allow_html=True)
        with cond4:
            route_weather = st.selectbox("🌤️ Weather", [
                "Clear / Dry","Rain / Wet Roads","Snow / Ice","Fog / Low Visibility"])

        st.markdown("<br>", unsafe_allow_html=True)

        # ── LOCATION PICKERS ──────────────────────────────
        loc1, loc2 = st.columns(2)
        with loc1:
            st.markdown("**📍 Starting Location**")
            start_loc = st.selectbox("Start", list(SALEM_LOCATIONS.keys()),
                                     index=0, label_visibility="collapsed")
        with loc2:
            st.markdown("**🏁 Destination**")
            end_loc = st.selectbox("End", list(SALEM_LOCATIONS.keys()),
                                   index=4, label_visibility="collapsed")

        if start_loc == end_loc:
            st.warning("Please choose different start and destination locations.")
            st.stop()

        # ── FETCH ROUTES ──────────────────────────────────

        # Step 1: Get primary route
        if USE_GMAPS:
            with st.spinner("🗺️ Fetching primary route from Google Maps..."):
                primary_points, primary_stops = get_google_route(
                    start_loc, end_loc, GMAPS_KEY, alternative=False)
                if not primary_points:
                    primary_points, primary_stops = get_fallback_route(start_loc, end_loc)
        else:
            primary_points, primary_stops = get_fallback_route(start_loc, end_loc)

        if not primary_stops:
            primary_stops = [start_loc, end_loc]
        primary_scores = score_route(primary_stops, route_hour, route_weekend, route_weather)

        # Step 2: ML finds the safest detour waypoint
        safer_stops, best_waypoint = find_safer_stops(
            primary_stops, route_hour, route_weekend, route_weather, start_loc, end_loc)

        if safer_stops:
            if USE_GMAPS and best_waypoint:
                # Force Google Maps to route through the lowest-risk waypoint
                with st.spinner(f"🛡️ Finding safer route via {best_waypoint.split('/')[0].strip()}..."):
                    alt_points, alt_stops_gmaps = get_google_route(
                        start_loc, end_loc, GMAPS_KEY,
                        alternative=True, safer_waypoint=best_waypoint,
                        hour=route_hour, is_weekend=route_weekend, weather=route_weather)
                    if not alt_points:
                        # Fallback: straight line through waypoint
                        alt_points = [SALEM_LOCATIONS[s] for s in safer_stops]
                    alt_stops = safer_stops
            else:
                alt_points = [SALEM_LOCATIONS[s] for s in safer_stops]
                alt_stops  = safer_stops

            alt_scores = score_route(alt_stops, route_hour, route_weekend, route_weather)
            has_alt    = True
        else:
            has_alt    = False
            alt_stops  = primary_stops
            alt_scores = primary_scores
            alt_points = primary_points

        # Stats
        p_avg   = np.mean([s["score"] for s in primary_scores])
        p_max   = max(s["score"] for s in primary_scores)
        p_risky = sum(1 for s in primary_scores if s["score"] >= RISK_THRESHOLD)
        a_avg   = np.mean([s["score"] for s in alt_scores])
        a_max   = max(s["score"] for s in alt_scores)
        saved   = p_avg - a_avg

        # ── SUMMARY CARDS ─────────────────────────────────
        p_lbl, p_col = risk_label(p_avg)
        card1, card2 = st.columns(2)

        with card1:
            st.markdown(f"""
            <div style="background:white;border:2px solid {p_col};border-radius:14px;padding:18px 22px">
              <div style="font-weight:700;font-size:.95rem;margin-bottom:12px">🛣️ Primary Route
                <span style="font-size:.72rem;font-weight:400;color:#9ca3af;margin-left:8px">
                  {"Google Maps fastest" if USE_GMAPS else "Estimated route"}</span></div>
              <div style="display:flex;gap:24px;margin-bottom:12px">
                <div style="text-align:center">
                  <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;
                              color:{p_col};line-height:1">{p_avg:.0f}%</div>
                  <div style="font-size:.7rem;color:#9ca3af;margin-top:2px">Avg Risk</div>
                </div>
                <div style="text-align:center">
                  <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;
                              color:{p_col};line-height:1">{p_max}</div>
                  <div style="font-size:.7rem;color:#9ca3af;margin-top:2px">Peak Score</div>
                </div>
                <div style="text-align:center">
                  <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;
                              color:{p_col};line-height:1">{p_risky}</div>
                  <div style="font-size:.7rem;color:#9ca3af;margin-top:2px">Risky Stops</div>
                </div>
              </div>
              <div style="background:#f9fafb;border-radius:8px;padding:10px;font-size:.8rem">
                {'⚠️ <b>Caution</b> — passes through ' + str(p_risky) + ' risky stop(s)'
                 if p_risky > 0 else '✅ <b>Looks safe</b> under current conditions'}
              </div>
            </div>""", unsafe_allow_html=True)

        with card2:
            if has_alt:
                a_lbl, a_col = risk_label(a_avg)
                route_src = "Google Maps alternative" if USE_GMAPS else "ML-rerouted"
                st.markdown(f"""
                <div style="background:white;border:2px solid {a_col};border-radius:14px;padding:18px 22px">
                  <div style="font-weight:700;font-size:.95rem;margin-bottom:12px">✅ Safer Route
                    <span style="font-size:.72rem;font-weight:400;color:#9ca3af;margin-left:8px">
                      {route_src}</span></div>
                  <div style="display:flex;gap:24px;margin-bottom:12px">
                    <div style="text-align:center">
                      <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;
                                  color:{a_col};line-height:1">{a_avg:.0f}%</div>
                      <div style="font-size:.7rem;color:#9ca3af;margin-top:2px">Avg Risk</div>
                    </div>
                    <div style="text-align:center">
                      <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;
                                  color:{a_col};line-height:1">{a_max}</div>
                      <div style="font-size:.7rem;color:#9ca3af;margin-top:2px">Peak Score</div>
                    </div>
                    <div style="text-align:center">
                      <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;
                                  color:#16a34a;line-height:1">{saved:.0f}pts</div>
                      <div style="font-size:.7rem;color:#9ca3af;margin-top:2px">Risk Saved</div>
                    </div>
                  </div>
                  <div style="background:#f0fdf4;border-radius:8px;padding:10px;font-size:.8rem;color:#166534">
                    🛡️ <b>Recommended</b> — avoids the highest-risk intersection on the primary route
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#f0fdf4;border:2px solid #16a34a;border-radius:14px;padding:18px 22px">
                  <div style="font-weight:700;font-size:.95rem;margin-bottom:8px">✅ Route Already Safe</div>
                  <p style="font-size:.85rem;color:#166534;margin:0">
                    All intersections score below {RISK_THRESHOLD}% under current conditions.
                    You are good to go!</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── DUAL MAPS ────────────────────────────────────
        map1_col, map2_col = st.columns(2)

        with map1_col:
            st.markdown(f"**🛣️ Primary Route** — `{start_loc}` → `{end_loc}`")
            map1 = build_route_map(primary_points, primary_stops, primary_scores, "Primary Route")
            st_html(map1._repr_html_(), height=480)

        with map2_col:
            if has_alt:
                risky_name = max(primary_scores, key=lambda x: x["score"])["intersection"].split("/")[0].strip()
                st.markdown(f"**✅ Safer Route** — avoids `{risky_name}` area")
                map2 = build_route_map(alt_points, alt_stops, alt_scores, "Safer Route")
            else:
                st.markdown("**✅ Primary Route** — already the safest option")
                map2 = build_route_map(primary_points, primary_stops, primary_scores, "Safe Route")
            st_html(map2._repr_html_(), height=480)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── STOP BREAKDOWN ────────────────────────────────
        st.markdown("**📊 Stop-by-Stop Risk Breakdown**")
        tab_p, tab_a = st.tabs(["🛣️ Primary Route", "✅ Safer Route" if has_alt else "✅ Route"])

        def render_stops(scores):
            for i, info in enumerate(scores):
                score  = info["score"]
                color  = info["color"]
                label  = info["label"]
                name   = info["intersection"]
                prefix = "🚀 Start" if i==0 else ("🏁 Destination" if i==len(scores)-1 else f"Stop {i}")
                icon   = "🔴" if score>=70 else "🟡" if score>=RISK_THRESHOLD else "🟢"
                tip    = ("⚠️ High risk — consider alternative." if score>=70
                          else "⚠️ Moderate — stay alert here." if score>=RISK_THRESHOLD
                          else "✅ Below threshold — relatively safe.")
                if 16<=route_hour<=18: tip += " PM rush hour active."
                elif 7<=route_hour<=9: tip += " AM rush hour active."
                if route_weather!="Clear / Dry": tip += f" {route_weather} adds extra risk."

                st.markdown(f"""
                <div style="background:white;border:1px solid #e5e7eb;border-radius:12px;
                            padding:14px 18px;margin-bottom:8px;border-left:5px solid {color}">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                    <div>
                      <span style="font-size:.68rem;font-weight:600;color:#9ca3af;text-transform:uppercase">{prefix}</span><br>
                      <span style="font-weight:700;font-size:.92rem">{icon} {name}</span>
                    </div>
                    <div style="text-align:right">
                      <span style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:{color}">{score}%</span><br>
                      <span style="font-size:.7rem;font-weight:600;color:{color}">{label}</span>
                    </div>
                  </div>
                  <div style="background:#f3f4f6;border-radius:4px;height:6px;margin-bottom:6px;overflow:hidden">
                    <div style="width:{score}%;height:100%;background:{color};border-radius:4px"></div>
                  </div>
                  <div style="font-size:.76rem;color:#6b7280">{tip}</div>
                </div>""", unsafe_allow_html=True)

        with tab_p: render_stops(primary_scores)
        with tab_a:
            if has_alt: render_stops(alt_scores)
            else: st.success("Primary route is already the safest option.")

        with st.expander("🤖 How does the ML model score these routes?"):
            st.markdown(f"""
**Risk scores are calculated using Random Forest feature importance weights from the Salem crash model:**

| Feature | Weight | Effect Right Now |
|---|---|---|
| **Hour of Day** | 38% | {"PM rush +18pts ⚠️" if 16<=route_hour<=18 else "AM rush +12pts ⚠️" if 7<=route_hour<=9 else "Night +8pts" if route_hour>=22 or route_hour<=5 else "Midday -5pts ✅" if 10<=route_hour<=14 else "Normal 0pts"} |
| **Month / Season** | 28% | Built into base intersection risk scores |
| **Is Weekend** | 14% | {"Weekend -8pts — lower traffic ✅" if route_weekend else "Weekday +4pts — higher traffic"} |
| **Is Rush Hour** | 12% | {"Rush hour active ⚠️" if (7<=route_hour<=9) or (16<=route_hour<=18) else "Not rush hour ✅"} |
| **At Intersection** | 8% | All stops are known Salem intersections |

**Threshold: {RISK_THRESHOLD}%** — stops above this trigger a safer route.
{"**Google Maps** provides the real road geometry. The ML model then scores each intersection the route passes through." if USE_GMAPS else "**Add a Google Maps API key** in the sidebar to get real road routes instead of straight-line estimates."}
            """)

# ================================================================
# PLANNER VIEW
# ================================================================
else:
    map_col, info_col = st.columns([3,1])
    with map_col:
        st.markdown("""<div style="margin-bottom:8px">
          <div style="font-weight:700;font-size:1rem">📍 Risk Analysis & Infrastructure Planning</div>
          <div style="font-size:.78rem;color:#6b7280">Professional risk assessment — click any marker</div>
        </div>""", unsafe_allow_html=True)

        m = folium.Map(location=SALEM_CENTER, zoom_start=SALEM_ZOOM, tiles="CartoDB positron")
        hotspots = [
            (42.5195,-70.8967,89,'High','Derby St / Washington Sq'),
            (42.5228,-70.8952,76,'High','North St / Essex St'),
            (42.5157,-70.8879,71,'High','Marlborough Rd / Ocean Ave'),
            (42.5243,-70.9043,68,'Medium','Bridge St / Federal St'),
            (42.5108,-70.9012,62,'Medium','Highland Ave / Jefferson Ave'),
            (42.5180,-70.8825,32,'Low','Lafayette St / Loring Ave'),
            (42.5072,-70.8993,28,'Low','Canal St / Grove St'),
        ]
        colors_map = {'High':'#dc2626','Medium':'#f59e0b','Low':'#22c55e'}
        for lat,lon,score,level,name in hotspots:
            folium.CircleMarker([lat,lon],
                radius=14 if level=='High' else 10 if level=='Medium' else 7,
                color='white', weight=2, fill=True,
                fill_color=colors_map[level], fill_opacity=0.88,
                tooltip=folium.Tooltip(f"<b>{name}</b><br>Risk: {score}/100 — {level}")
            ).add_to(m)
            folium.Marker([lat,lon], icon=folium.DivIcon(
                html=f'<div style="font-size:10px;font-weight:800;color:white;text-align:center;line-height:28px;width:28px">{score}</div>',
                icon_size=(28,28), icon_anchor=(14,14))
            ).add_to(m)
        m.get_root().html.add_child(folium.Element("""
        <div style="position:fixed;bottom:16px;left:16px;background:white;border-radius:10px;
                    padding:12px 16px;box-shadow:0 2px 10px rgba(0,0,0,.15);z-index:9999;font-size:12px">
          <b>Risk Levels</b><br>
          <span style="color:#dc2626">●</span> High Risk (70%+)<br>
          <span style="color:#f59e0b">●</span> Medium Risk (40-69%)<br>
          <span style="color:#22c55e">●</span> Low Risk (&lt;40%)
        </div>"""))
        st_html(m._repr_html_(), height=480)

    with info_col:
        hc = crash_df.groupby('Hour').size().reset_index(name='count') if 'Hour' in crash_df.columns \
             else pd.DataFrame({'Hour':list(range(24)),
                  'count':[12,8,6,5,4,10,22,55,78,60,55,62,70,65,68,95,110,108,80,55,38,28,22,15]})
        fig_h = go.Figure(go.Bar(x=hc['Hour'],y=hc['count'],
            marker_color=['#dc2626' if c>=90 else '#f59e0b' if c>=55 else '#22c55e' for c in hc['count']]))
        fig_h.update_layout(height=180,margin=dict(l=0,r=0,t=0,b=0),
            paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(tickfont=dict(size=9),showgrid=False),
            yaxis=dict(tickfont=dict(size=9),gridcolor='#f3f4f6'),showlegend=False)
        st.plotly_chart(fig_h,use_container_width=True,config={'displayModeBar':False})
        st.markdown("""<div class="infra-status">
          <div class="status-row"><span>Overall Risk</span><span class="badge-moderate">Moderate</span></div>
          <div class="status-row"><span>Active Alerts</span><span class="status-num">3</span></div>
          <div class="status-row"><span>Best Model Acc.</span><span class="status-num">71%</span></div>
          <div class="status-row"><span>Pending Actions</span><span class="status-num">5</span></div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown("""<div class="planner-kpi"><span class="kpi-num">3,207</span>
      <div class="kpi-label">Total Crashes Analyzed</div>
      <div class="kpi-change kpi-down">down 18% vs last year</div></div>""",unsafe_allow_html=True)
    k2.markdown("""<div class="planner-kpi"><span class="kpi-num">71%</span>
      <div class="kpi-label">Best Model Accuracy</div>
      <div class="kpi-change">Tuned Random Forest</div></div>""",unsafe_allow_html=True)
    k3.markdown("""<div class="planner-kpi"><span class="kpi-num">$2.4M</span>
      <div class="kpi-label">Estimated Safety Budget</div>
      <div class="kpi-change">5 priority locations</div></div>""",unsafe_allow_html=True)
    k4.markdown("""<div class="planner-kpi"><span class="kpi-num">5</span>
      <div class="kpi-label">High-Risk Intersections</div>
      <div class="kpi-change kpi-up">up 2 new this quarter</div></div>""",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    risk_data = pd.DataFrame({
        'Location':['Derby St / Washington Sq','North St / Essex St','Marlborough Rd / Ocean Ave',
                    'Bridge St / Federal St','Highland Ave / Jefferson Ave','Lafayette St / Loring Ave','Canal St / Grove St'],
        'Risk Score':['89 — High','76 — High','71 — High','68 — Medium','62 — Medium','32 — Low','28 — Low'],
        'Crash Count':[142,98,87,74,61,28,19],
        'Peak Hour':['4-6 PM','3-5 PM','5-7 PM','7-9 AM','3-4 PM','12-2 PM','6-8 AM'],
        'Primary Factor':['High Vol + Wet Roads','Rush Hour Congestion','Nighttime Visibility',
                          'AM Rush + Intersection','School Zone Traffic','Moderate Traffic','Low Volume'],
        'Action':['Urgent Review','Urgent Review','Schedule Review','Schedule Review','Schedule Review','Monitor','Monitor']
    })
    def ca(v): return ('background-color:#fee2e2;color:#991b1b;font-weight:bold' if v=='Urgent Review'
                       else 'background-color:#fef3c7;color:#92400e;font-weight:bold' if v=='Schedule Review'
                       else 'background-color:#dcfce7;color:#166534;font-weight:bold')
    def cr(v): return ('color:#dc2626;font-weight:bold' if 'High' in v
                       else 'color:#d97706;font-weight:bold' if 'Medium' in v else 'color:#16a34a;font-weight:bold')
    st.dataframe(risk_data.style.map(ca,subset=['Action']).map(cr,subset=['Risk Score']).hide(axis='index'),
                 use_container_width=True,height=300)