# NEBULA X — Track 3: Autonomous Train Door Telemetry Diagnostics & Predictive Maintenance

> **Team Name:** The Out-liars  
> **Track:** Track 3 – Predictive Fault Detection & Diagnostics for Transit Systems  
> **Status:** Production-Ready MVP & Google Cloud Run Deployed 🚀

---

## 🔗 Quick Links
* **Live App Dashboard:** https://nebulax-app-435023706723.us-central1.run.app/
* **Pitch Video:** https://youtu.be/f__jzvHL-8s
* **Final Validation Score:** **0.842** (IoU-Weighted F1 Score)

---

## 📌 Executive Summary
In high-density transit networks like Singapore’s MRT, a single train door failure during peak operational hours triggers severe line congestion, commuter delays, and steep financial reliability penalties exceeding **$22,500 per major incident**. 

**NEBULA X** is an end-to-end intelligent diagnostic system that shifts maintenance from reactive repairs to predictive prevention. By combining unsupervised anomaly detection with explainable supervised rules, NEBULA X flags faulty door cycles before failure occurs while providing transparent, human-readable logic for depot engineers.

---

## 🏗️ Two-Stage Hybrid Machine Learning Architecture

1. **Stage 1 (Unsupervised Baseline):** Profiles normal, healthy door cycles using K-Means clustering across core telemetry streams to compute a continuous Euclidean anomaly distance score.
2. **Stage 2 (Supervised Classification):** Balances minority fault classes using **SMOTE** and trains a depth-bounded **Decision Tree (`max_depth=8`)** optimized for high fault recall (`84.9%`) to eliminate dangerous false negatives.

---

## 🚀 Key Features & Streamlit Dashboard

Our interactive operator dashboard (`app.py`) features:
* **🎛️ Live Telemetry Simulation:** Sidebar sensor sliders allowing operators to simulate real-time motor currents, voltages, and forces.
* **📊 Door Motor Telemetry Tab:** Interactive Plotly scatter clusters visualizing healthy vs. abnormal resistance boundaries.
* **📈 Model Evaluation Tab:** Rigorous quantitative validation metrics (82.7% accuracy, confusion matrix, precision/recall audits on an 80/20 holdout test set).
* **🌲 Explainable Rules Tab:** Visual decision tree mapping with custom confidence-based color mapping and full 8-level rule logic export.
* **📄 Automated Submission Pipeline:** Generates official hackathon-compliant prediction logs (`door_predictions.csv`).

---

## 📂 Project Structure

```tree
NEBULAX/
├── .streamlit/            # UI configuration for the dashboard
├── assets/                # Visual assets (Header banner & architecture diagram)
├── app.py                 # Main Streamlit Operator Dashboard
├── pipeline.py            # ML pipeline (K-Means, SMOTE, Decision Tree)
├── generate_preds.py      # Segmentation logic that groups predictions (Score: 0.84)
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container configuration for Google Cloud Run
├── .dockerignore          # Optimization filters for deployment
├── NEBULA_X_Presentation.pptx # Pitch deck presentation slides
├── door_predictions.csv   # Raw output logs
├── predictions.zip        # Final hackathon-compliant submission file
└── README.md              # Project documentation
