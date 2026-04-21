import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_store.feature_definitions import FeatureGenerator
from feature_store.offline_store import OfflineFeatureStore
from training.train import ModelTrainer
from monitoring.drift_detector import DriftDetector

st.set_page_config(
    page_title="ML Feature Store",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ ML Feature Store + Model Serving Platform")
st.markdown("**Feature Store → MLflow Training → Model Registry → Drift Monitoring**")
st.divider()


# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Controls")

    n_customers = st.slider("Number of Customers", 500, 2000, 1000)
    run_btn = st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True)

    st.divider()
    st.markdown("""
    **Pipeline Steps:**
    1. Generate customer data
    2. Compute ML features
    3. Store in Feature Store
    4. Train 3 models with MLflow
    5. Compare model performance
    6. Detect data drift

    **[View on GitHub](https://github.com/saimanjunathk/ml-feature-store)**
    """)


# ── Run Pipeline ──
if run_btn:
    with st.spinner("🔧 Generating data and computing features..."):
        gen      = FeatureGenerator(n_customers=n_customers)
        raw_df   = gen.generate_raw_data()
        feat_df  = gen.compute_features(raw_df)

        store = OfflineFeatureStore()
        store.save_features(feat_df)

        X, y = store.load_features(
            feature_columns=gen.FEATURE_COLUMNS,
            target_column=gen.TARGET_COLUMN
        )

    with st.spinner("🤖 Training models with MLflow..."):
        trainer = ModelTrainer()
        results = trainer.train_all_models(X, y)

    with st.spinner("📊 Running drift detection..."):
        detector = DriftDetector()
        detector.set_reference(feat_df[gen.FEATURE_COLUMNS])
        drifted_df = detector.simulate_drift(feat_df[gen.FEATURE_COLUMNS], drift_magnitude=0.3)
        drift_report = detector.detect_drift(drifted_df)

    st.session_state["raw_df"]       = raw_df
    st.session_state["feat_df"]      = feat_df
    st.session_state["results"]      = results
    st.session_state["drift_report"] = drift_report
    st.session_state["trainer"]      = trainer
    st.session_state["X"]            = X
    st.session_state["y"]            = y


if "results" not in st.session_state:
    st.info("👈 Click **Run Full Pipeline** to start!")
    st.stop()


# ── Load from session state ──
raw_df       = st.session_state["raw_df"]
feat_df      = st.session_state["feat_df"]
results      = st.session_state["results"]
drift_report = st.session_state["drift_report"]
trainer      = st.session_state["trainer"]
X            = st.session_state["X"]
y            = st.session_state["y"]


# ── Tabs ──
tab1, tab2, tab3, tab4 = st.tabs([
    "📦 Feature Store",
    "🤖 Model Training",
    "🏆 Model Comparison",
    "📡 Drift Monitoring"
])


# ─────────────────────────────────────────────
# TAB 1: FEATURE STORE
# ─────────────────────────────────────────────
with tab1:
    st.subheader("📦 Feature Store")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👥 Total Customers", f"{len(raw_df):,}")
    with col2:
        st.metric("🔢 Total Features", len(X.columns))
    with col3:
        st.metric("🎯 Churn Rate", f"{y.mean()*100:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Feature Distributions**")
        feature = st.selectbox("Select Feature", X.columns.tolist())
        fig = px.histogram(
            feat_df, x=feature, color="churned",
            nbins=30,
            color_discrete_map={0: "#00d4ff", 1: "#ef4444"},
            labels={"churned": "Churned"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Churn by Plan**")
        churn_plan = feat_df.groupby("plan")["churned"].mean().reset_index()
        fig = px.bar(
            churn_plan, x="plan", y="churned",
            color="churned",
            labels={"churned": "Churn Rate", "plan": "Plan"}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Feature Store Sample**")
    st.dataframe(feat_df[X.columns.tolist() + ["churned"]].head(10),
                hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 2: MODEL TRAINING
# ─────────────────────────────────────────────
with tab2:
    st.subheader("🤖 MLflow Experiment Tracking")

    # Show MLflow runs
    runs_df = trainer.get_experiment_runs()

    if not runs_df.empty:
        st.markdown("**All Experiment Runs**")

        # Select relevant columns
        display_cols = [
            c for c in [
                "tags.mlflow.runName",
                "params.model_type",
                "metrics.accuracy",
                "metrics.auc",
                "metrics.f1_score",
                "start_time"
            ] if c in runs_df.columns
        ]
        st.dataframe(
            runs_df[display_cols].head(20),
            hide_index=True,
            use_container_width=True
        )

    st.divider()

    # Show best model details
    best = results[0]
    st.markdown(f"**Best Model: {best['model_type'].upper()}**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Accuracy", f"{best['accuracy']*100:.2f}%")
    with col2:
        st.metric("📈 AUC Score", f"{best['auc']:.4f}")
    with col3:
        st.metric("⚖️ F1 Score", f"{best['f1']:.4f}")

    # Feature importance
    if hasattr(best["model"], "feature_importances_"):
        st.markdown("**Feature Importance**")
        importance_df = pd.DataFrame({
            "feature":    best["feature_names"],
            "importance": best["model"].feature_importances_
        }).sort_values("importance", ascending=False)

        fig = px.bar(
            importance_df.head(10),
            x="importance", y="feature",
            orientation="h", color="importance"
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 3: MODEL COMPARISON
# ─────────────────────────────────────────────
with tab3:
    st.subheader("🏆 Model Comparison")

    comparison_df = pd.DataFrame([{
        "Model":    r["model_type"],
        "Accuracy": round(r["accuracy"]*100, 2),
        "AUC":      round(r["auc"], 4),
        "F1 Score": round(r["f1"], 4)
    } for r in results])

    st.dataframe(comparison_df, hide_index=True, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            comparison_df,
            x="Model", y="AUC",
            color="Model",
            title="AUC Score by Model"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            comparison_df,
            x="Model", y="Accuracy",
            color="Model",
            title="Accuracy by Model"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ROC curves
    st.markdown("**ROC Curves**")
    fig = go.Figure()

    from sklearn.metrics import roc_curve
    colors = ["#00d4ff", "#7c3aed", "#10b981"]

    for i, result in enumerate(results):
        fpr, tpr, _ = roc_curve(result["y_test"], result["y_pred_prob"])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"{result['model_type']} (AUC={result['auc']:.3f})",
            line=dict(color=colors[i], width=2)
        ))

    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1],
        line=dict(dash="dash", color="gray"),
        name="Random"
    ))

    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        title="ROC Curves — All Models"
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 4: DRIFT MONITORING
# ─────────────────────────────────────────────
with tab4:
    st.subheader("📡 Data Drift Monitoring")
    st.caption("Comparing training data distribution vs simulated production data")

    n_drifted = drift_report["drifted"].sum()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📊 Features Monitored", len(drift_report))
    with col2:
        st.metric(
            "🚨 Features Drifted",
            n_drifted,
            delta=f"{n_drifted/len(drift_report)*100:.0f}% of features",
            delta_color="inverse"
        )
    with col3:
        avg_psi = drift_report["psi"].mean()
        st.metric(
            "📈 Avg PSI Score",
            f"{avg_psi:.4f}",
            delta="High drift!" if avg_psi > 0.2 else "Low drift"
        )

    st.divider()

    # Drift report table
    st.markdown("**Feature Drift Report**")
    styled = drift_report.copy()
    st.dataframe(
        styled,
        hide_index=True,
        use_container_width=True,
        column_config={
            "drifted": st.column_config.CheckboxColumn("Drifted?"),
            "psi": st.column_config.NumberColumn("PSI", format="%.4f"),
            "mean_change": st.column_config.NumberColumn("Mean Change %", format="%.2f%%")
        }
    )

    # PSI bar chart
    fig = px.bar(
        drift_report.sort_values("psi", ascending=False),
        x="feature", y="psi",
        color="drifted",
        color_discrete_map={True: "#ef4444", False: "#10b981"},
        title="PSI Score by Feature (Red = Drifted)",
        labels={"psi": "PSI Score", "feature": "Feature"}
    )
    fig.add_hline(y=0.2, line_dash="dash", line_color="orange",
                  annotation_text="Drift Threshold (0.2)")
    st.plotly_chart(fig, use_container_width=True)