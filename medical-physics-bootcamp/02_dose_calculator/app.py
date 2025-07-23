"""
Streamlit front-end for the radiation dose calculator.
Run with:
    streamlit run app.py
"""

import streamlit as st
from dose_calculator import calculate_dose, safety_check

st.set_page_config(page_title="Radiation Dose Calculator", page_icon="⚛️")

st.title("⚛️ Radiation Dose Calculator")

dose_rate = st.slider("Dose rate (Gy/min)", 0.0, 10.0, 1.0, 0.1)
time_min  = st.number_input("Exposure time (min)", 0.0, 60.0, 1.0)

total = calculate_dose(dose_rate, time_min)
st.metric("Total dose (Gy)", f"{total:0.2f}")

msg, status = safety_check(total, 2.0)
if status == "warning":
    st.error(msg)
else:
    st.success(msg)

st.caption(
    "1 Gy = 1 J/kg absorbed. Threshold chosen for demo only — "
    "consult clinical protocols for real limits."
)
