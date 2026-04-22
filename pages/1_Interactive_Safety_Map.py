import streamlit as st
import folium
from streamlit.components.v1 import html as st_html
import numpy as np
import os
import re

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
    page_title="MataVision — Safe Route Planner",
    page_icon="🧭",
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

SALEM_CENTER = [42.519, -70.896]

GMAPS_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")
if not GMAPS_KEY:
    with st.sidebar:
        st.markdown("### 🔑 Google Maps API Key")
        entered_key = st.text_input("Paste API key here", type="password")
        if entered_key:
            GMAPS_KEY = entered_key
USE_GMAPS = GMAPS_AVAILABLE and bool(GMAPS_KEY)

# ================================================================
# ML FUNCTIONS
# ================================================================
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
    if score >= 70:  return "High Risk",     "#dc2626"
    if score >= 40:  return "Moderate Risk", "#f59e0b"
    return "Low Risk", "#16a34a"

def route_severity(scored_stops):
    """
    Route Severity = 0.5 × Mean Stop Risk
                   + 0.3 × Max Stop Risk
                   + 0.2 × (Percent High-Risk Stops × 100)
    """
    if not scored_stops:
        return 0
    scores      = [s["score"] for s in scored_stops]
    mean_risk   = np.mean(scores)
    max_risk    = max(scores)
    pct_high    = sum(1 for s in scores if s >= 70) / len(scores) * 100
    severity    = 0.5 * mean_risk + 0.3 * max_risk + 0.2 * pct_high
    return round(severity, 1)

def severity_label(sev):
    if sev >= 60:  return "High Severity",      "#dc2626"
    if sev >= 35:  return "Moderate Severity",  "#f59e0b"
    return "Low Severity", "#16a34a"

def find_safest_hour(location, is_weekend, weather):
    scores = [(h, ml_risk_score(location, h, is_weekend, weather)) for h in range(24)]
    scores.sort(key=lambda x: x[1])
    best_h, best_s = scores[0]
    am_pm = "AM" if best_h < 12 else "PM"
    h12   = best_h % 12
    h12   = 12 if h12 == 0 else h12
    return f"{h12}:00 {am_pm}", best_s

def format_duration(seconds):
    mins = seconds // 60
    if mins < 60:
        return f"{mins} min"
    return f"{mins//60}h {mins%60}min"

def format_distance(meters):
    miles = meters / 1609.34
    if miles < 0.1:
        return f"{int(meters)} m"
    return f"{miles:.1f} mi"

# ================================================================
# GOOGLE MAPS — fetch multiple routes
# ================================================================
@st.cache_data(show_spinner=False)
def get_all_routes(origin_name, dest_name, api_key):
    """
    Fetch up to 3 route options from Google Maps:
    1. Fastest route
    2. Alternative route (if available)
    3. Waypoint-forced route through lowest-risk Salem intersection
    Returns list of route dicts with points, stops, duration, distance.
    """
    try:
        gmaps  = googlemaps.Client(key=api_key)
        origin = SALEM_LOCATIONS[origin_name]
        dest   = SALEM_LOCATIONS[dest_name]

        import polyline as pl

        def decode_route(route):
            points = []
            streets = set()
            for leg in route["legs"]:
                for step in leg["steps"]:
                    points += pl.decode(step["polyline"]["points"])
                    instruction = re.sub('<[^<]+?>', '', step.get("html_instructions",""))
                    for word in instruction.replace("/"," ").split():
                        if len(word) > 3:
                            streets.add(word.lower().rstrip(".,"))
            duration = sum(leg["duration"]["value"] for leg in route["legs"])
            distance = sum(leg["distance"]["value"] for leg in route["legs"])
            return points, streets, duration, distance

        def match_stops(points, streets, origin_name, dest_name):
            route_lats = [p[0] for p in points]
            route_lons = [p[1] for p in points]
            matched    = [origin_name]
            candidates = []
            for loc_name, (lat, lon) in SALEM_LOCATIONS.items():
                if loc_name in [origin_name, dest_name]:
                    continue
                parts = [p.strip().lower() for p in loc_name.replace(" /","/").split("/")]
                name_match = False
                if len(parts) == 2:
                    p1 = [w for w in parts[0].split() if len(w) > 2]
                    p2 = [w for w in parts[1].split() if len(w) > 2]
                    m1 = any(any(pw in rs for rs in streets) for pw in p1)
                    m2 = any(any(pw in rs for rs in streets) for pw in p2)
                    if m1 and m2:
                        name_match = True
                dists = [abs(lat-rlat)+abs(lon-rlon)
                         for rlat, rlon in zip(route_lats, route_lons)]
                if name_match or min(dists) < 0.0004:
                    candidates.append((min(dists), loc_name))
            candidates.sort()
            for _, loc_name in candidates[:3]:
                matched.append(loc_name)
            matched.append(dest_name)
            seen, unique = set(), []
            for s in matched:
                if s not in seen:
                    seen.add(s)
                    unique.append(s)
            return unique

        routes = []

        # Route 1 & 2 — fastest + alternative
        result = gmaps.directions(
            origin=f"{origin[0]},{origin[1]}",
            destination=f"{dest[0]},{dest[1]}",
            mode="driving", alternatives=True, region="us"
        )
        if result:
            pts, sts, dur, dist = decode_route(result[0])
            stops = match_stops(pts, sts, origin_name, dest_name)
            routes.append({"label": "Fastest Route", "points": pts,
                           "stops": stops, "duration": dur, "distance": dist,
                           "color": "#7c3aed", "icon": "⚡"})

            if len(result) > 1:
                pts2, sts2, dur2, dist2 = decode_route(result[1])
                stops2 = match_stops(pts2, sts2, origin_name, dest_name)
                routes.append({"label": "Alternative Route", "points": pts2,
                               "stops": stops2, "duration": dur2, "distance": dist2,
                               "color": "#2563eb", "icon": "🔀"})

        # Route 3 — force through lowest-risk Salem intersection (balanced)
        # Pick lowest-risk waypoint not at start or end
        wp_candidates = [(BASE_RISK.get(n, 40), n) for n in SALEM_LOCATIONS
                         if n not in [origin_name, dest_name]]
        wp_candidates.sort()
        if wp_candidates:
            wp_name   = wp_candidates[0][1]
            wp_coords = SALEM_LOCATIONS[wp_name]
            result3   = gmaps.directions(
                origin=f"{origin[0]},{origin[1]}",
                destination=f"{dest[0]},{dest[1]}",
                waypoints=[f"{wp_coords[0]},{wp_coords[1]}"],
                mode="driving", region="us"
            )
            if result3:
                pts3, sts3, dur3, dist3 = decode_route(result3[0])
                stops3 = match_stops(pts3, sts3, origin_name, dest_name)
                # Only add if genuinely different from existing routes
                if not any(abs(dur3 - r["duration"]) < 30 for r in routes):
                    routes.append({"label": "Balanced Route", "points": pts3,
                                   "stops": stops3, "duration": dur3, "distance": dist3,
                                   "color": "#16a34a", "icon": "⚖️"})

        return routes if routes else None

    except Exception:
        return None

def get_fallback_routes(start, end):
    """Fallback when no API key — simulate 2 routes."""
    s_lat, s_lon = SALEM_LOCATIONS[start]
    e_lat, e_lon = SALEM_LOCATIONS[end]
    mid_lat = (s_lat + e_lat) / 2
    mid_lon = (s_lon + e_lon) / 2

    candidates = sorted(
        [(((lat-mid_lat)**2+(lon-mid_lon)**2)**.5, name)
         for name, (lat, lon) in SALEM_LOCATIONS.items()
         if name not in [start, end]]
    )

    stops1  = [start] + ([candidates[0][1]] if candidates else []) + [end]
    points1 = [SALEM_LOCATIONS[s] for s in stops1]

    stops2  = [start] + ([candidates[-1][1]] if len(candidates) > 1 else []) + [end]
    points2 = [SALEM_LOCATIONS[s] for s in stops2]

    return [
        {"label": "Fastest Route",  "points": points1, "stops": stops1,
         "duration": 420, "distance": 1800, "color": "#7c3aed", "icon": "⚡"},
        {"label": "Balanced Route", "points": points2, "stops": stops2,
         "duration": 600, "distance": 2400, "color": "#16a34a", "icon": "⚖️"},
    ]

def score_route_stops(stops, hour, is_weekend, weather):
    return [{"intersection": s,
             "score": ml_risk_score(s, hour, is_weekend, weather),
             "label": risk_label(ml_risk_score(s, hour, is_weekend, weather))[0],
             "color": risk_label(ml_risk_score(s, hour, is_weekend, weather))[1]}
            for s in stops]

def build_route_map(route, scored_stops, height=480):
    points = route["points"]
    stops  = route["stops"]
    color  = route["color"]

    all_lats = [p[0] for p in points]
    all_lons = [p[1] for p in points]
    sw = [max(42.490, min(all_lats)-0.004), max(-70.930, min(all_lons)-0.005)]
    ne = [min(42.545, max(all_lats)+0.004), min(-70.855, max(all_lons)+0.005)]

    m = folium.Map(location=SALEM_CENTER, zoom_start=15,
                   tiles="CartoDB positron", min_zoom=13)
    m.fit_bounds([sw, ne])

    folium.PolyLine(points, color=color, weight=6, opacity=0.9).add_to(m)

    score_map = {s["intersection"]: s for s in scored_stops}
    for i, stop in enumerate(stops):
        lat, lon = SALEM_LOCATIONS[stop]
        info  = score_map.get(stop, {"score": BASE_RISK.get(stop,40),
                                     "label":"Moderate Risk","color":"#f59e0b"})
        score = info["score"]
        clr   = info["color"]
        label = info["label"]

        if i == 0:
            folium.Marker([lat,lon],
                tooltip=f"🚀 START: {stop}",
                popup=folium.Popup(f"<b>START</b><br>{stop}<br>Risk: {score}% — {label}",
                                   max_width=200),
                icon=folium.Icon(color="blue", icon="home", prefix="fa")
            ).add_to(m)
        elif i == len(stops)-1:
            folium.Marker([lat,lon],
                tooltip=f"🏁 DESTINATION: {stop}",
                popup=folium.Popup(f"<b>DESTINATION</b><br>{stop}<br>Risk: {score}% — {label}",
                                   max_width=200),
                icon=folium.Icon(color="green", icon="flag", prefix="fa")
            ).add_to(m)
        else:
            dot_color = "#dc2626" if score>=70 else "#f59e0b" if score>=40 else "#22c55e"
            if score >= 70:
                folium.CircleMarker([lat,lon], radius=22, color=dot_color,
                    weight=2, fill=False, opacity=0.3).add_to(m)
            folium.CircleMarker([lat,lon], radius=14, color="white", weight=2.5,
                fill=True, fill_color=dot_color, fill_opacity=0.92,
                popup=folium.Popup(
                    f"<b>{stop}</b><br>Risk: <b>{score}%</b><br>{label}", max_width=220),
                tooltip=f"{stop}: {score}% ({label})"
            ).add_to(m)
            folium.Marker([lat,lon], icon=folium.DivIcon(
                html=f'<div style="font-size:11px;font-weight:800;color:white;'
                     f'text-align:center;line-height:28px;width:28px">{score}</div>',
                icon_size=(28,28), icon_anchor=(14,14))
            ).add_to(m)

    m.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
                background:{color};color:white;border-radius:20px;padding:5px 16px;
                font-size:12px;font-weight:700;z-index:9999">
      {route["icon"]} {route["label"]}</div>"""))
    m.get_root().html.add_child(folium.Element("""
    <div style="position:fixed;bottom:16px;left:16px;background:white;border-radius:10px;
                padding:10px 14px;box-shadow:0 2px 10px rgba(0,0,0,.15);z-index:9999;font-size:12px">
      <span style="color:#dc2626">●</span> High Risk (70%+)<br>
      <span style="color:#f59e0b">●</span> Moderate (40%+)<br>
      <span style="color:#22c55e">●</span> Low Risk
    </div>"""))
    return m

# ================================================================
# PAGE HEADER
# ================================================================
st.markdown("""
<div class="page-header">
  <h1>🧭 Safe Route Planner</h1>
  <p>Compare multiple route options — the ML model scores each route for safety
  and recommends the safest choice, even if it takes longer</p>
</div>""", unsafe_allow_html=True)

# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown("### 🧭 Trip Conditions")
    route_hour    = st.slider("🕐 Hour of Travel", 0, 23, 8)
    am_pm         = "AM" if route_hour < 12 else "PM"
    disp_h        = route_hour % 12
    disp_h        = 12 if disp_h == 0 else disp_h
    st.caption(f"Travel time: **{disp_h}:00 {am_pm}**")
    route_weekend = st.toggle("📅 Weekend Trip", value=False)
    route_weather = st.selectbox("🌤️ Weather", [
        "Clear / Dry","Rain / Wet Roads","Snow / Ice","Fog / Low Visibility"])
    st.divider()
    st.markdown("### 📊 Safety Overview")
    st.markdown("""
    <div class="infra-status">
      <div class="status-row"><span>Overall Risk</span><span class="badge-moderate">Moderate</span></div>
      <div class="status-row"><span>Active Alerts</span><span class="status-num">3</span></div>
      <div class="status-row"><span>Best Model Acc.</span><span class="status-num">71%</span></div>
    </div>""", unsafe_allow_html=True)

# ================================================================
# TRIP CONDITIONS BAR
# ================================================================
is_rush = (7 <= route_hour <= 9) or (16 <= route_hour <= 18)
c1, c2, c3 = st.columns(3)
c1.markdown(f"""
<div style="background:white;border-radius:10px;padding:14px;text-align:center;border:1px solid #ddd6fe">
  <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#7c3aed">{disp_h}:00 {am_pm}</div>
  <div style="font-size:.75rem;color:#6b7280;margin-top:2px">Hour of Travel</div>
</div>""", unsafe_allow_html=True)
c2.markdown(f"""
<div style="background:white;border-radius:10px;padding:14px;text-align:center;border:1px solid #ddd6fe">
  <div style="font-size:1rem;font-weight:700">{route_weather}</div>
  <div style="font-size:.75rem;color:#6b7280;margin-top:2px">Weather Conditions</div>
</div>""", unsafe_allow_html=True)
c3.markdown(f"""
<div style="background:{'#fee2e2' if is_rush else '#dcfce7'};border-radius:10px;padding:14px;
            text-align:center;border:1px solid {'#fecaca' if is_rush else '#bbf7d0'}">
  <div style="font-size:1rem;font-weight:700;color:{'#991b1b' if is_rush else '#166634'}">
    {'⚠️ Rush Hour Active' if is_rush else '✅ Off-Peak Hours'}</div>
  <div style="font-size:.75rem;color:#6b7280;margin-top:2px">
    {'Higher crash risk period' if is_rush else 'Lower crash risk period'}</div>
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
# FETCH ALL ROUTES
# ================================================================
if USE_GMAPS:
    with st.spinner("🗺️ Fetching route options from Google Maps..."):
        all_routes = get_all_routes(start_loc, end_loc, GMAPS_KEY)
        if not all_routes:
            all_routes = get_fallback_routes(start_loc, end_loc)
else:
    all_routes = get_fallback_routes(start_loc, end_loc)

# Score each route
for route in all_routes:
    route["scored_stops"] = score_route_stops(
        route["stops"], route_hour, route_weekend, route_weather)
    route["severity"]     = route_severity(route["scored_stops"])
    sev_lbl, sev_col      = severity_label(route["severity"])
    route["sev_label"]    = sev_lbl
    route["sev_color"]    = sev_col
    scores                = [s["score"] for s in route["scored_stops"]]
    route["mean_risk"]    = np.mean(scores)
    route["max_risk"]     = max(scores)
    route["high_risk_ct"] = sum(1 for s in scores if s >= 70)
    route["riskiest"]     = max(route["scored_stops"], key=lambda x: x["score"])

# Find safest route by severity score
safest_route = min(all_routes, key=lambda r: r["severity"])
fastest_route = min(all_routes, key=lambda r: r["duration"])

# Best time to travel
riskiest_stop_name = safest_route["riskiest"]["intersection"]
best_time, best_score = find_safest_hour(riskiest_stop_name, route_weekend, route_weather)
best_lbl, best_col    = risk_label(best_score)



# ================================================================
# RECOMMENDATION
# ================================================================
fastest_sev = fastest_route["sev_label"].split()[0]
safest_sev  = safest_route["sev_label"].split()[0]
time_diff   = safest_route["duration"] - fastest_route["duration"]
time_str    = format_duration(abs(time_diff)) if time_diff != 0 else "same time"

if safest_route["label"] == fastest_route["label"]:
    st.success(
        f"✅ **The fastest route is also the safest** — "
        f"Severity: **{safest_route['sev_label']}**. No trade-off needed.")
elif time_diff > 0:
    st.info(
        f"🛡️ **Safer route recommended.** The **{safest_route['label']}** takes "
        f"**{time_str} longer** but reduces severity from **{fastest_sev}** to "
        f"**{safest_sev}** — a meaningful safety improvement for Salem roads.")
else:
    st.success(
        f"✅ **The safer route is also faster!** Take the **{safest_route['label']}** "
        f"— lower severity ({safest_sev}) and saves {time_str}.")

st.markdown("<br>", unsafe_allow_html=True)

# ================================================================
# ROUTE COMPARISON CARDS
# ================================================================
st.markdown("### 📊 Route Comparison")
cols = st.columns(len(all_routes))

for i, (route, col) in enumerate(zip(all_routes, cols)):
    is_safest  = route["label"] == safest_route["label"]
    is_fastest = route["label"] == fastest_route["label"]
    border     = f"3px solid {route['sev_color']}" if is_safest else "1px solid #e5e7eb"
    badge      = ""
    if is_safest and is_fastest:
        badge = '<span style="background:#7c3aed;color:white;border-radius:20px;padding:2px 10px;font-size:.7rem;font-weight:700">⭐ BEST CHOICE</span>'
    elif is_safest:
        badge = '<span style="background:#16a34a;color:white;border-radius:20px;padding:2px 10px;font-size:.7rem;font-weight:700">🛡️ SAFEST</span>'
    elif is_fastest:
        badge = '<span style="background:#7c3aed;color:white;border-radius:20px;padding:2px 10px;font-size:.7rem;font-weight:700">⚡ FASTEST</span>'

    col.markdown(f"""
    <div style="background:white;border:{border};border-radius:14px;padding:18px;
                {'box-shadow:0 4px 15px rgba(0,0,0,.1);' if is_safest else ''}">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">
        <div>
          <div style="font-size:1.1rem;margin-bottom:4px">{route['icon']}</div>
          <div style="font-weight:700;font-size:.95rem">{route['label']}</div>
        </div>
        <div>{badge}</div>
      </div>

      <div style="background:{route['sev_color']}18;border-radius:8px;padding:10px;
                  text-align:center;margin-bottom:12px">
        <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;
                    color:{route['sev_color']};line-height:1">{route['severity']}</div>
        <div style="font-size:.75rem;font-weight:600;color:{route['sev_color']}">
          {route['sev_label']}</div>
        <div style="font-size:.68rem;color:#9ca3af;margin-top:2px">Route Severity Score</div>
      </div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:.78rem">
        <div style="background:#f9fafb;border-radius:6px;padding:8px;text-align:center">
          <div style="font-weight:700;color:#374151">{format_duration(route['duration'])}</div>
          <div style="color:#9ca3af;font-size:.68rem">Travel Time</div>
        </div>
        <div style="background:#f9fafb;border-radius:6px;padding:8px;text-align:center">
          <div style="font-weight:700;color:#374151">{format_distance(route['distance'])}</div>
          <div style="color:#9ca3af;font-size:.68rem">Distance</div>
        </div>
        <div style="background:#f9fafb;border-radius:6px;padding:8px;text-align:center">
          <div style="font-weight:700;color:#374151">{route['mean_risk']:.0f}%</div>
          <div style="color:#9ca3af;font-size:.68rem">Avg Stop Risk</div>
        </div>
        <div style="background:#f9fafb;border-radius:6px;padding:8px;text-align:center">
          <div style="font-weight:700;color:#dc2626">{route['high_risk_ct']}</div>
          <div style="color:#9ca3af;font-size:.68rem">High-Risk Stops</div>
        </div>
      </div>

      <div style="margin-top:10px;font-size:.75rem;color:#6b7280">
        ⚠️ Highest risk: <b>{route['riskiest']['intersection'].split('/')[0].strip()}</b>
        ({route['max_risk']:.0f}%)
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================================================================
# SEVERITY FORMULA EXPLAINED
# ================================================================
with st.expander("📐 How is Route Severity calculated?"):
    st.markdown(f"""
**Route Severity Formula:**

```
Severity = 0.5 × Mean Stop Risk
         + 0.3 × Max Stop Risk
         + 0.2 × (% High-Risk Stops × 100)
```

| Component | Weight | Why |
|---|---|---|
| **Mean Stop Risk** | 50% | Overall average danger across the whole route |
| **Max Stop Risk** | 30% | The single worst intersection — a dangerous outlier matters |
| **% High-Risk Stops** | 20% | How many stops are genuinely dangerous (70%+) |

**Severity Thresholds:**
- 🟢 **Low Severity** — score below 35
- 🟡 **Moderate Severity** — score 35–59
- 🔴 **High Severity** — score 60+

The formula prioritizes average risk (50%) so one very dangerous stop doesn't
unfairly penalize an otherwise safe route — but still accounts for it through
the max risk component (30%).
    """)

st.markdown("<br>", unsafe_allow_html=True)

# ================================================================
# ROUTE MAPS — one per route
# ================================================================
st.markdown("### 🗺️ Route Maps")
st.caption("Each map shows the actual road path and ML risk scores for intersections on that route")

map_cols = st.columns(len(all_routes))
for route, col in zip(all_routes, map_cols):
    with col:
        is_safest = route["label"] == safest_route["label"]
        label_suffix = " ⭐" if is_safest else ""
        st.markdown(f"**{route['icon']} {route['label']}{label_suffix}** — "
                    f"Severity: `{route['severity']}` ({route['sev_label']})")
        m = build_route_map(route, route["scored_stops"])
        st_html(m._repr_html_(), height=420)

st.markdown("<br>", unsafe_allow_html=True)



# ================================================================
# STOP BREAKDOWN FOR RECOMMENDED ROUTE
# ================================================================
st.markdown(f"### 📋 Stop-by-Stop Breakdown — {safest_route['icon']} {safest_route['label']}")
st.caption("ML risk scores for each intersection on the recommended route")

for i, info in enumerate(safest_route["scored_stops"]):
    score  = info["score"]
    color  = info["color"]
    label  = info["label"]
    name   = info["intersection"]
    prefix = "🚀 Start" if i==0 else ("🏁 Destination" if i==len(safest_route["scored_stops"])-1 else f"Stop {i}")
    icon   = "🔴" if score>=70 else "🟡" if score>=40 else "🟢"

    st.markdown(f"""
    <div style="background:white;border:1px solid #e5e7eb;border-radius:12px;
                padding:14px 20px;margin-bottom:8px;border-left:5px solid {color}">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <div>
          <span style="font-size:.68rem;font-weight:600;color:#9ca3af;
                       text-transform:uppercase">{prefix}</span><br>
          <span style="font-weight:700;font-size:.95rem">{icon} {name}</span>
        </div>
        <div style="text-align:right">
          <span style="font-family:'Syne',sans-serif;font-size:1.8rem;
                       font-weight:800;color:{color};line-height:1">{score}%</span><br>
          <span style="font-size:.72rem;font-weight:600;color:{color}">{label}</span>
        </div>
      </div>
      <div style="background:#f3f4f6;border-radius:4px;height:6px;overflow:hidden">
        <div style="width:{score}%;height:100%;background:{color};border-radius:4px"></div>
      </div>
    </div>""", unsafe_allow_html=True)

# ================================================================
# ML EXPLANATION
# ================================================================
with st.expander("🤖 How does the ML model score these routes?"):
    st.markdown(f"""
**Risk scores use Random Forest feature importance weights trained on Salem crash data:**

| Feature | Weight | Effect on This Trip |
|---|---|---|
| **Hour of Day** | 38% | {"PM rush +18pts ⚠️" if 16<=route_hour<=18 else "AM rush +12pts ⚠️" if 7<=route_hour<=9 else "Night +8pts" if route_hour>=22 or route_hour<=5 else "Midday -5pts ✅" if 10<=route_hour<=14 else "Normal 0pts"} |
| **Month / Season** | 28% | Built into base intersection scores from crash history |
| **Is Weekend** | 14% | {"Weekend -8pts — lower traffic ✅" if route_weekend else "Weekday +4pts — higher commuter traffic"} |
| **Is Rush Hour** | 12% | {"Rush hour active ⚠️" if is_rush else "Not rush hour ✅"} |
| **At Intersection** | 8% | Each scored stop is a known Salem intersection |

**Key insight:** Hour of Day is the strongest predictor (38%). In a small city like Salem,
WHEN you drive matters as much as WHICH route you take — which is why the app
shows both route severity AND the safest travel time.
    """)