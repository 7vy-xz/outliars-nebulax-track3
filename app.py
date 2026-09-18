import streamlit as st
import pandas as pd
import plotly.express as px
import os
from pipeline import train_and_evaluate_pipeline, predict_realtime_sample, generate_submission_csv
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import re

st.set_page_config(page_title="NEBULA X - LTA Door Maintenance", layout="wide")

# --- SIDEBAR SETUP ---
with st.sidebar:
    # 1. Put the header banner right at the top of the sidebar
    if os.path.exists("assets/header_banner.png"):
        st.image("assets/header_banner.png", use_container_width=True)
    
    st.markdown("---")
    st.header("Live Train Telemetry Sensors")
    
    # 2. Group all your sidebar sliders neatly inside the sidebar block
    motor_curr = st.slider("Motor Current (mA)", 100.0, 3000.0, 120.0)
    motor_voltage = st.slider("Motor Voltage (10mV)", 100.0, 2000.0, 450.0)
    motor_force = st.slider("Motor Electrodynamic Force", 0.0, 100.0, 50.0)
    open_time = st.slider("Door Opening Time (.1s)", 10.0, 100.0, 24.0)
    close_time = st.slider("Door Closing Time (.1s)", 10.0, 100.0, 24.0)

# --- MAIN PAGE CONTENT ---
# Custom CSS to prevent metric value text truncation
st.markdown("""
<style>
div[data-testid="stMetricValue"] {
    font-size: 22px !important;
    white-space: normal !important;
    word-break: break-word !important;
}
</style>
""", unsafe_allow_html=True)

df, kmeans, centroid, dt_model, rules_text, features, eval_metrics = train_and_evaluate_pipeline()

st.title("NEBULA X: LTA Train Door Diagnostic System")
st.caption("Track 3: Predictive Fault Detection | Autonomous Door Telemetry Diagnostics")

st.divider()

sample = {
    'Motor current(mA)': motor_curr,
    'Motor Voltage(10mV)': motor_voltage,
    'Motor electrodynamic force': motor_force,
    'Door opening time(.1s)': open_time,
    'Door closing time(.1s)': close_time
}

score, fault_pred = predict_realtime_sample(sample, centroid, dt_model, features)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Anomaly Distance Score", f"{score:.2f}")
with col2:
    if fault_pred == "Normal":
        st.success("STATUS: Healthy Cycle")
    else:
        st.error(f"CRITICAL: {fault_pred}")
with col3:
    st.metric("Est. Delay Cost Savings", f"${0 if fault_pred == 'Normal' else 22500:,}")
with col4:
    st.metric("Diagnosed State", fault_pred)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Door Motor Telemetry", 
    "📈 Model Evaluation",
    "🌲 Explainable Rules", 
    "🏗️ Pipeline Architecture"
])


with tab1:
    st.subheader("Motor Current vs Electrodynamic Force")
    fig = px.scatter(
        df, x="Motor current(mA)", y="Motor electrodynamic force", 
        color="status", title="LTA Train Door Telemetry Clusters",
        color_discrete_map={"Normal": "#059669", "Abnormal resistance": "#FF1744"},
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Quantitative Model Validation (80/20 Holdout Test)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Accuracy", f"{eval_metrics['accuracy']*100:.1f}%")
    
    rep = eval_metrics['report']
    abnormal_key = 'Abnormal resistance' if 'Abnormal resistance' in rep else list(rep.keys())[0]
    m2.metric("Precision (Faults)", f"{rep[abnormal_key]['precision']*100:.1f}%")
    m3.metric("Recall (Faults)", f"{rep[abnormal_key]['recall']*100:.1f}%")
    m4.metric("F1-Score", f"{rep[abnormal_key]['f1-score']*100:.1f}%")
    
    st.markdown("#### Confusion Matrix")
    cm_fig = px.imshow(
        eval_metrics['confusion_matrix'],
        x=list(eval_metrics['classes']),
        y=list(eval_metrics['classes']),
        text_auto=True,
        color_continuous_scale='Blues',
        labels=dict(x="Predicted Label", y="True Label", color="Count")
    )
    st.plotly_chart(cm_fig, use_container_width=True)

with tab3:
    st.subheader("Visual Decision Tree Rules")
    
    # 1. Add a clean, modern Legend using Streamlit HTML
    st.markdown("""
    <div style='display: flex; gap: 20px; padding: 12px; background-color: #f8f9fa; border-radius: 8px; margin-bottom: 20px; border: 1px solid #e9ecef;'>
        <div><span style='display: inline-block; width: 16px; height: 16px; background-color: rgba(220, 53, 69, 0.8); margin-right: 6px; border-radius: 3px;'></span> <b>Abnormal (Fault)</b></div>
        <div><span style='display: inline-block; width: 16px; height: 16px; background-color: rgba(25, 135, 84, 0.8); margin-right: 6px; border-radius: 3px;'></span> <b>Normal (Healthy)</b></div>
        <div><span style='display: inline-block; width: 16px; height: 16px; background-color: #6c757d; margin-right: 6px; border-radius: 3px;'></span> <b>(...)</b> Truncated Branches</div>
        <div style='margin-left: auto; color: #495057;'><i>* Darker color intensity = Higher statistical confidence</i></div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Generate the visual tree
    fig, ax = plt.subplots(figsize=(24, 8), dpi=300)
    annotations = plot_tree(
        dt_model,
        feature_names=features,
        class_names=[str(c) for c in dt_model.classes_],
        filled=True,
        rounded=True,
        max_depth=2,
        fontsize=12,
        ax=ax,
        impurity=False,
        proportion=True,
        precision=1
    )
    
    # 3. Dynamically Recolor the Nodes to match Streamlit UI
    for ann in annotations:
        text = ann.get_text()
        box = ann.get_bbox_patch()
        
        # Regex to extract the probability values: value = [Abnormal_Prob, Normal_Prob]
        match = re.search(r"value = \[([\d\.]+),\s*([\d\.]+)\]", text)
        
        if box is not None:
            if "class = Normal" in text:
                alpha = float(match.group(2)) if match else 0.5
                box.set_facecolor((0.1, 0.6, 0.2, alpha))  # Green matching st.success
            elif "class = Abnormal" in text:
                alpha = float(match.group(1)) if match else 0.5
                box.set_facecolor((0.8, 0.1, 0.2, alpha))  # Red matching st.error
            else:
                # Style the (...) nodes if box exists
                box.set_facecolor("#6c757d") 
                ann.set_color("white")
        elif "(...)" in text:
            # Fallback to draw a box if Matplotlib skipped it
            ann.set_bbox(dict(facecolor="#6c757d", edgecolor="black", boxstyle="round,pad=0.2"))
            ann.set_color("white")

    st.pyplot(fig, clear_figure=True)
    
    # Keep the expander for the full 8-level audit
    with st.expander("🔍 View Full 8-Level Decision Logic (Advanced)"):
        st.markdown("For deep technical auditing, here is the complete extracted rule logic:")
        st.code(rules_text, language="text")

with tab4:
    st.subheader("System Architecture & Pipeline Blueprint")
    st.markdown("Two-stage hybrid machine learning architecture combining unsupervised anomaly detection with explainable rules.")
    
    if os.path.exists("assets/pipeline_diagram.png"):
        st.image("assets/pipeline_diagram.png", use_container_width=True, caption="NEBULA X Track 3 Architecture Flow")
    else:
        st.info("Pipeline diagram image not found in assets folder. Please ensure 'pipeline_diagram.png' is saved inside the 'assets/' directory.")