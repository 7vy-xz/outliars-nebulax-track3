import streamlit as st
import pandas as pd
import plotly.express as px
import os
from pipeline import train_pipeline, predict_realtime_sample, generate_submission_csv

st.set_page_config(page_title="NEBULA X - LTA Door Maintenance", layout="wide")

# Load Pipeline trained on Official LTA Data
df, kmeans, centroid, dt_model, rules_text, features = train_pipeline()

# 1. Header Banner
if os.path.exists("assets/header_banner.png"):
    st.image("assets/header_banner.png", use_container_width=True)
else:
    st.title("🚇 NEBULA X: LTA Train Door Diagnostic System")
    st.caption("Track 3: Predictive Fault Detection | Autonomous Door Telemetry Diagnostics")

st.divider()

# 2. Real-Time Telemetry Fault Injector
st.sidebar.header("🎛️ Live Train Telemetry Sensors")
st.sidebar.markdown("Adjust sensors to simulate live train door operations:")

motor_curr = st.sidebar.slider("Motor Current (mA)", 50, 2500, 120, 10)
motor_volt = st.sidebar.slider("Motor Voltage (10mV)", 200, 1000, 450, 20)
motor_force = st.sidebar.slider("Motor Electrodynamic Force", 10, 200, 50, 5)
open_time = st.sidebar.slider("Door Opening Time (.1s)", 10, 60, 24, 1)
close_time = st.sidebar.slider("Door Closing Time (.1s)", 10, 60, 24, 1)

sample = {
    'Motor current(mA)': motor_curr,
    'Motor Voltage(10mV)': motor_volt,
    'Motor electrodynamic force': motor_force,
    'Door opening time(.1s)': open_time,
    'Door closing time(.1s)': close_time
}

# Live Real-time Prediction
score, fault_pred = predict_realtime_sample(sample, centroid, dt_model, features)

# Top Key Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Anomaly Distance Score", value=f"{score:.2f}")

with col2:
    if fault_pred == "Normal":
        st.success("STATUS: Healthy Cycle")
    else:
        st.error(f"CRITICAL: {fault_pred}")

with col3:
    saved = 0 if fault_pred == "Normal" else 22500
    st.metric(label="Est. Delay Cost Savings", value=f"${saved:,}")

with col4:
    st.metric(label="Diagnosed State", value=fault_pred)

st.divider()

# Visualizations & Submission Generator
tab1, tab2, tab3 = st.tabs(["📊 Door Motor Telemetry", "🌲 Explainable Rules", "📄 Export Submission CSV"])

with tab1:
    st.subheader("Motor Current vs Electrodynamic Force")
    fig = px.scatter(
        df, x="Motor current(mA)", y="Motor electrodynamic force", 
        color="status", title="LTA Train Door Telemetry Clusters",
        color_discrete_map={"Normal": "#00E676", "Abnormal": "#FF1744"}
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Human-Readable Decision Tree Rules")
    st.code(rules_text, language="text")

with tab3:
    st.subheader("Generate Official Hackathon Submission File")
    st.markdown("Fulfills **Submission Requirement #4** (`door_predictions.csv`).")
    if st.button("🚀 Export Predictions CSV"):
        out_path = generate_submission_csv()
        st.success(f"Successfully generated file at `{out_path}`!")