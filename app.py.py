import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="MataVision",
    page_icon="🚗",
    layout="wide"
)

# --- Apply CSS ---
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

col1, col2 = st.columns([1,4])
with col1:
    st.image("assets/logo.png", width=80)
with col2:
    st.markdown("## **MataVision** – Predicting Risks, Protecting Lives")
    st.write("Machine learning predictions to prevent crashes and save lives in **Salem, Massachusetts**.")

st.markdown("---")

# --- Navigation Buttons ---
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🚦 Interactive Safety Map"):
        st.switch_page("pages/1_Interactive_Safety_Map.py")
with col2:
    if st.button("🏗 City Planner Dashboard"):
        st.switch_page("pages/2_City_Planner_Dashboard.py")
with col3:
    if st.button("💹 Insurance Analytics"):
        st.switch_page("pages/3_Insurance_Analytics.py")

st.markdown("---")

