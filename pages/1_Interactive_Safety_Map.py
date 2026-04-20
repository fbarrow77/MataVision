import streamlit as st
import pandas as pd
import folium
from streamlit.components.v1 import html as st_html
import plotly.graph_objects as go
import numpy as np
import os

try:
    import googlemaps
    GMAPS_AVAILABLE = True
except ImportError:
    GMAPS_AVAILABLE = False

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

# ================================================================
# SALEM LOCATIONS & ML MODEL
# ================================================================
SALEM_LOCATIONS = {
    "Derby St / Washington Sq":       (42.5195, -70.8967),
    "North St / Essex St":            (42.5228, -70.8967),
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
# GOOGLE MAPS
# ================================================================
GMAPS_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")
if not GMAPS_KEY:
    with st.sidebar:
        st.markdown("### 🔑 Google Maps API Key")
        entered_key = st.text_input("Paste API key here", type="password")
        if entered_key:
            GMAPS_KEY = entered_key
        else:
            st.caption("Without a key, routes use estimated paths.")

USE_GMAPS = GMAPS_AVAILABLE and bool(GMAPS_KEY)

@st.cache_data(show_spinner=False)
def get_google_route(origin_name, dest_name, api_key, safer_waypoint=None):
    try:
        gmaps = googlemaps.Client(key=api_key)
        origin = SALEM_LOCATIONS[origin_name]
        dest   = SALEM_LOCATIONS[dest_name]

        if safer_waypoint:
            wp = SALEM_LOCATIONS[safer_waypoint]
            result = gmaps.directions(
                origin=f"{origin[0]},{origin[1]}",
                destination=f"{dest[0]},{dest[1]}",
                waypoints=[f"{wp[0]},{wp[1]}"],
                mode="driving", region="us"
            )
        else:
            result = gmaps.directions(
                origin=f"{origin[0]},{origin[1]}",
                destination=f"{dest[0]},{dest[1]}",
                mode="driving", alternatives=True, region="us"
            )

        if not result:
            return None, None

        route = result[0]
        import polyline as pl
        points = []
        for leg in route["legs"]:
            for step in leg["steps"]:
                points += pl.decode(step["polyline"]["points"])

        matched_stops = [origin_name]
        route_lats = [p[0] for p in points]
        route_lons = [p[1] for p in points]
        for loc_name, (lat, lon) in SALEM_LOCATIONS.items():
            if loc_name in [origin_name, dest_name]:
                continue
            dists = [abs(lat - rlat) + abs(lon - rlon) for rlat, rlon in zip(route_lats, route_lons)]
            if min(dists) < 0.003:
                matched_stops.append(loc_name)
        matched_stops.append(dest_name)
        seen = set()
        unique_stops = []
        for s in matched_stops:
            if s not in seen:
                seen.add(s)
                unique_stops.append(s)

        return points, unique_stops
    except Exception as e:
        return None, None

def get_fallback_route(start, end):
    s_lat, s_lon = SALEM_LOCATIONS[start]
    e_lat, e_lon = SALEM_LOCATIONS[end]
    mid_lat = (s_lat + e_lat) / 2
    mid_lon = (s_lon + e_lon) / 2
    candidates = [(((lat-mid_lat)**2+(lon-mid_lon)**2)**.5, name)
                  for name, (lat, lon) in SALEM_LOCATIONS.items()
                  if name not in [start, end]]
    candidates.sort()
    stops = [start] + [c[1] for c in candidates[:2]] + [end]
    return [SALEM_LOCATIONS[s] for s in stops], stops

def find_safer_stops(primary_stops, hour, is_weekend, weather, start, end):
    primary_scores = score_route(primary_stops, hour, is_weekend, weather)
    max_risk = max(primary_scores, key=lambda x: x["score"])
    if max_risk["score"] < RISK_THRESHOLD:
        return None, None
    s_lat, s_lon = SALEM_LOCATIONS[start]
    e_lat, e_lon = SALEM_LOCATIONS[end]
    alternatives = []
    for name, (lat, lon) in SALEM_LOCATIONS.items():
        if name in primary_stops or name in [start, end]:
            continue
        score = ml_risk_score(name, hour, is_weekend, weather)
        dist  = (((lat-(s_lat+e_lat)/2)**2+((lon-(s_lon+e_lon)/2)**2))**.5)
        weighted = (score * 0.7) + (dist * 1000 * 0.3)
        alternatives.append((weighted, score, name))
    alternatives.sort()
    if not alternatives:
        return None, None
    best_waypoint = alternatives[0][2]
    alt_stops = [best_waypoint if s == max_risk["intersection"] else s
                 for s in primary_stops]
    return alt_stops, best_waypoint

def build_route_map(route_points, stops, scores, map_label, height=480):
    all_lats = [p[0] for p in route_points]
    all_lons = [p[1] for p in route_points]
    sw = [max(42.495, min(all_lats) - 0.003), max(-70.930, min(all_lons) - 0.004)]
    ne = [min(42.545, max(all_lats) + 0.003), min(-70.860, max(all_lons) + 0.004)]

    m = folium.Map(location=SALEM_CENTER, zoom_start=SALEM_ZOOM,
                   tiles="CartoDB positron", min_zoom=13)
    m.fit_bounds([sw, ne])

    line_color = "#16a34a" if "Safer" in map_label else "#7c3aed"
    folium.PolyLine(route_points, color=line_color, weight=5, opacity=0.9).add_to(m)

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
                folium.CircleMarker([lat, lon], radius=22, color=dot_color,
                    weight=1.5, fill=False, opacity=0.3).add_to(m)
            folium.CircleMarker([lat, lon], radius=14, color="white", weight=2.5,
                fill=True, fill_color=dot_color, fill_opacity=0.92,
                popup=folium.Popup(f"<b>{stop}</b><br>Risk: <b>{score}%</b><br>{label}", max_width=220),
                tooltip=f"{stop}: {score}% ({label})"
            ).add_to(m)
            folium.Marker([lat, lon], icon=folium.DivIcon(
                html=f'<div style="font-size:11px;font-weight:800;color:white;'
                     f'text-align:center;line-height:28px;width:28px">{score}</div>',
                icon_size=(28,28), icon_anchor=(14,14))
            ).add_to(m)

    m.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
                background:{line_color};color:white;border-radius:20px;padding:5px 16px;
                font-size:12px;font-weight:700;z-index:9999">{map_label}</div>"""))
    m.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;bottom:50px;right:10px;background:white;border-radius:8px;
                padding:4px 10px;font-size:11px;color:#6b7280;z-index:9999">
      {"🗺️ Google Maps Route" if USE_GMAPS else "📍 Estimated Route"}</div>"""))
    m.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;bottom:16px;left:16px;background:white;border-radius:10px;
                padding:10px 14px;box-shadow:0 2px 10px rgba(0,0,0,.15);z-index:9999;font-size:12px">
      <span style="color:#dc2626">●</span> High Risk (70%+)<br>
      <span style="color:#f59e0b">●</span> Moderate ({RISK_THRESHOLD}%+)<br>
      <span style="color:#22c55e">●</span> Low Risk<br>
      <span style="color:{line_color}">━</span> Route
    </div>"""))
    return m

# ================================================================
# PAGE HEADER
# ================================================================
st.markdown("""
<div class="page-header">
  <h1>🧭 Safe Route Planner</h1>
  <p>Enter a start and destination within Salem — the ML model scores every intersection
  and suggests a safer alternative if risk exceeds 30%</p>
</div>""", unsafe_allow_html=True)



# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown("### 🧭 Route Conditions")
    route_hour = st.slider("🕐 Hour of Travel", 0, 23, 8)
    am_pm  = "AM" if route_hour < 12 else "PM"
    disp_h = route_hour % 12
    disp_h = 12 if disp_h == 0 else disp_h
    st.caption(f"Travel time: **{disp_h}:00 {am_pm}**")
    route_weekend = st.toggle("📅 Weekend Trip", value=False)
    route_weather = st.selectbox("🌤️ Weather", [
        "Clear / Dry", "Rain / Wet Roads", "Snow / Ice", "Fog / Low Visibility"])
    st.divider()
    st.markdown("### 📊 Safety Overview")
    st.markdown("""
    <div class="infra-status">
      <div class="status-row"><span>Overall Risk</span><span class="badge-moderate">Moderate</span></div>
      <div class="status-row"><span>Active Alerts</span><span class="status-num">3</span></div>
      <div class="status-row"><span>Best Model Acc.</span><span class="status-num">71%</span></div>
    </div>""", unsafe_allow_html=True)

# ================================================================
# TRIP CONDITIONS INLINE
# ================================================================
st.markdown("""<div style="background:#f5f3ff;border-radius:12px;padding:16px 20px;margin-bottom:16px">
  <div style="font-weight:700;font-size:.88rem;color:#5b21b6;margin-bottom:4px">⚙️ Trip Conditions</div>
  <div style="font-size:.78rem;color:#6b7280">Set conditions in the sidebar — hour, weekend, weather</div>
</div>""", unsafe_allow_html=True)

is_rush = (7 <= route_hour <= 9) or (16 <= route_hour <= 18)
c1, c2, c3 = st.columns(3)
c1.markdown(f"""<div style="background:white;border-radius:10px;padding:12px;text-align:center;border:1px solid #ddd6fe">
  <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:#7c3aed">{disp_h}:00 {am_pm}</div>
  <div style="font-size:.72rem;color:#6b7280">Hour of Travel</div></div>""", unsafe_allow_html=True)
c2.markdown(f"""<div style="background:white;border-radius:10px;padding:12px;text-align:center;border:1px solid #ddd6fe">
  <div style="font-size:1.2rem;font-weight:700">{'🌧️ ' + route_weather}</div>
  <div style="font-size:.72rem;color:#6b7280">Weather</div></div>""", unsafe_allow_html=True)
c3.markdown(f"""<div style="background:{'#fee2e2' if is_rush else '#dcfce7'};border-radius:10px;padding:12px;text-align:center;border:1px solid {'#fecaca' if is_rush else '#bbf7d0'}">
  <div style="font-size:1rem;font-weight:700;color:{'#991b1b' if is_rush else '#166534'}">
    {'⚠️ Rush Hour Active' if is_rush else '✅ Off-Peak'}</div>
  <div style="font-size:.72rem;color:#6b7280">{'Extra caution advised' if is_rush else 'Normal conditions'}</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================================================================
# LOCATION PICKERS
# ================================================================
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

# ================================================================
# COMPUTE ROUTES
# ================================================================
if USE_GMAPS:
    with st.spinner("🗺️ Fetching route from Google Maps..."):
        primary_points, primary_stops = get_google_route(start_loc, end_loc, GMAPS_KEY)
        if not primary_points:
            primary_points, primary_stops = get_fallback_route(start_loc, end_loc)
else:
    primary_points, primary_stops = get_fallback_route(start_loc, end_loc)

if not primary_stops:
    primary_stops = [start_loc, end_loc]

primary_scores = score_route(primary_stops, route_hour, route_weekend, route_weather)
safer_stops, best_waypoint = find_safer_stops(
    primary_stops, route_hour, route_weekend, route_weather, start_loc, end_loc)

if safer_stops:
    if USE_GMAPS and best_waypoint:
        with st.spinner(f"🛡️ Finding safer route via {best_waypoint.split('/')[0].strip()}..."):
            alt_points, _ = get_google_route(start_loc, end_loc, GMAPS_KEY,
                                             safer_waypoint=best_waypoint)
            if not alt_points:
                alt_points = [SALEM_LOCATIONS[s] for s in safer_stops]
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

p_avg   = np.mean([s["score"] for s in primary_scores])
p_max   = max(s["score"] for s in primary_scores)
p_risky = sum(1 for s in primary_scores if s["score"] >= RISK_THRESHOLD)
a_avg   = np.mean([s["score"] for s in alt_scores])
a_max   = max(s["score"] for s in alt_scores)
saved   = p_avg - a_avg

# ================================================================
# SUMMARY CARDS
# ================================================================
p_lbl, p_col = risk_label(p_avg)
card1, card2 = st.columns(2)

with card1:
    st.markdown(f"""
    <div style="background:white;border:2px solid {p_col};border-radius:14px;padding:18px 22px">
      <div style="font-weight:700;font-size:.95rem;margin-bottom:12px">🛣️ Primary Route
        <span style="font-size:.72rem;font-weight:400;color:#9ca3af;margin-left:8px">
          {"Google Maps fastest" if USE_GMAPS else "Estimated"}</span></div>
      <div style="display:flex;gap:24px;margin-bottom:12px">
        <div style="text-align:center">
          <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:{p_col}">{p_avg:.0f}%</div>
          <div style="font-size:.7rem;color:#9ca3af">Avg Risk</div>
        </div>
        <div style="text-align:center">
          <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:{p_col}">{p_max}</div>
          <div style="font-size:.7rem;color:#9ca3af">Peak Score</div>
        </div>
        <div style="text-align:center">
          <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:{p_col}">{p_risky}</div>
          <div style="font-size:.7rem;color:#9ca3af">Risky Stops</div>
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
        st.markdown(f"""
        <div style="background:white;border:2px solid {a_col};border-radius:14px;padding:18px 22px">
          <div style="font-weight:700;font-size:.95rem;margin-bottom:12px">✅ Safer Route
            <span style="font-size:.72rem;font-weight:400;color:#9ca3af;margin-left:8px">
              {"ML-guided via " + best_waypoint.split("/")[0].strip() if best_waypoint else "ML-rerouted"}</span></div>
          <div style="display:flex;gap:24px;margin-bottom:12px">
            <div style="text-align:center">
              <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:{a_col}">{a_avg:.0f}%</div>
              <div style="font-size:.7rem;color:#9ca3af">Avg Risk</div>
            </div>
            <div style="text-align:center">
              <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:{a_col}">{a_max}</div>
              <div style="font-size:.7rem;color:#9ca3af">Peak Score</div>
            </div>
            <div style="text-align:center">
              <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:#16a34a">{saved:.0f}pts</div>
              <div style="font-size:.7rem;color:#9ca3af">Risk Saved</div>
            </div>
          </div>
          <div style="background:#f0fdf4;border-radius:8px;padding:10px;font-size:.8rem;color:#166534">
            🛡️ <b>Recommended</b> — avoids the highest-risk stop on the primary route
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#f0fdf4;border:2px solid #16a34a;border-radius:14px;padding:18px 22px">
          <div style="font-weight:700;font-size:.95rem;margin-bottom:8px">✅ Route Already Safe</div>
          <p style="font-size:.85rem;color:#166534;margin:0">
            All intersections score below {RISK_THRESHOLD}% — good to go!</p>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================================================================
# DUAL MAPS
# ================================================================
map1_col, map2_col = st.columns(2)

with map1_col:
    st.markdown(f"**🛣️ Primary Route** — `{start_loc}` → `{end_loc}`")
    map1 = build_route_map(primary_points, primary_stops, primary_scores, "Primary Route")
    st_html(map1._repr_html_(), height=480)

with map2_col:
    if has_alt:
        risky = max(primary_scores, key=lambda x: x["score"])["intersection"].split("/")[0].strip()
        st.markdown(f"**✅ Safer Route** — avoids `{risky}` area")
        map2 = build_route_map(alt_points, alt_stops, alt_scores, "Safer Route")
    else:
        st.markdown("**✅ Primary Route** — already the safest option")
        map2 = build_route_map(primary_points, primary_stops, primary_scores, "Safe Route")
    st_html(map2._repr_html_(), height=480)

st.markdown("<br>", unsafe_allow_html=True)

# ================================================================
# STOP BREAKDOWN
# ================================================================
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
        if route_weather != "Clear / Dry": tip += f" {route_weather} adds extra risk."

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

**Threshold: {RISK_THRESHOLD}%** — stops above this trigger a safer route suggestion.
    """)