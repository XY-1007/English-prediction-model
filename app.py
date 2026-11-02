import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# ----------------- 1. Page Configuration -----------------
st.set_page_config(
    page_title="Postoperative Frailty Risk Calculator | HCC",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- 2. Model & Asset Caching -----------------
MODEL_DIR = "stacking_models"


@st.cache_resource
def load_all_artifacts():
    base_names = ['RF', 'CatBoost', 'ET']
    base_models = {name: joblib.load(os.path.join(MODEL_DIR, f"{name}.pkl")) for name in base_names}
    meta_learner = joblib.load(os.path.join(MODEL_DIR, "meta_learner.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

    # Load pre-computed explainer
    explainer_path = os.path.join(MODEL_DIR, "et_explainer.pkl")
    if os.path.exists(explainer_path):
        explainer = joblib.load(explainer_path)
    else:
        explainer = shap.TreeExplainer(base_models['ET'])

    # Load training data statistics
    summary_path = os.path.join(MODEL_DIR, "data_summary.pkl")
    data_summary = joblib.load(summary_path) if os.path.exists(summary_path) else None

    return base_models, meta_learner, feature_names, explainer, data_summary


base_models, meta_learner, feature_names, explainer, data_summary = load_all_artifacts()

# ----------------- 3. Header & Context -----------------
st.title("🩺 Online Clinical Decision Support System: Postoperative Heterogeneous Frailty Risk")
st.markdown("""
This platform utilizes a **Stacking Ensemble Learning Architecture (Random Forest + CatBoost + Extra Trees $\\rightarrow$ Logistic Regression)** 
to deliver real-time risk stratification and individualized explainability for postoperative heterogeneous frailty (`TrajStage`) in hepatocellular carcinoma (HCC) patients.
""")
st.markdown("---")

# ----------------- 4. Sidebar: Feature Input -----------------
st.sidebar.header("📋 Clinical Feature Input")
st.sidebar.markdown("Specify the patient's baseline and perioperative variables:")

user_input = {}

for feat in feature_names:
    if data_summary:
        f_min = float(data_summary['min'].get(feat, 0.0))
        f_max = float(data_summary['max'].get(feat, 100.0))
        f_default = float(data_summary['median'].get(feat, (f_min + f_max) / 2))
    else:
        f_min, f_max, f_default = 0.0, 100.0, 0.0

    # Auto-detect binary indicator variables (0/1)
    if f_min == 0.0 and f_max == 1.0:
        val = st.sidebar.selectbox(
            label=f"{feat}",
            options=[0, 1],
            index=int(f_default) if f_default in [0, 1] else 0,
            help="0: No / Negative / Absent | 1: Yes / Positive / Present"
        )
    else:
        step = 0.1 if (f_max - f_min) <= 10 else 1.0
        val = st.sidebar.number_input(
            label=f"{feat}",
            min_value=f_min,
            max_value=f_max,
            value=f_default,
            step=step
        )
    user_input[feat] = val

predict_button = st.sidebar.button("🚀 Calculate Risk & Explainability", type="primary", use_container_width=True)

# ----------------- 5. Prediction & SHAP Decomposition -----------------
col_pred, col_shap = st.columns([1, 1.2], gap="large")

if predict_button:
    input_df = pd.DataFrame([user_input])[feature_names]

    # Base learners step
    base_preds = np.zeros((1, len(base_models)))
    base_prob_dict = {}
    for i, (name, model) in enumerate(base_models.items()):
        prob = model.predict_proba(input_df)[0, 1]
        base_preds[0, i] = prob
        base_prob_dict[name] = prob

    # Meta-learner integration
    final_risk_prob = meta_learner.predict_proba(base_preds)[0, 1]

    with col_pred:
        st.subheader("📊 Risk Stratification Outcome")

        # Risk thresholds (tune to your cohort's optimal Youden Index)
        if final_risk_prob < 0.35:
            risk_cat = "Low Risk"
            badge_color = "success"
            rec_text = "The patient falls within the low-risk trajectory. Maintain standard Enhanced Recovery After Surgery (ERAS) protocols."
        elif final_risk_prob < 0.65:
            risk_cat = "Moderate Risk"
            badge_color = "warning"
            rec_text = "The patient exhibits moderate vulnerability. Close postoperative monitoring of serum albumin, electrolytes, and functional mobilization is advised."
        else:
            risk_cat = "High Risk"
            badge_color = "error"
            rec_text = "The patient is at high risk for heterogeneous frailty. Immediate pre-/postoperative multimodal prehabilitation and targeted nutritional support are strongly recommended."

        # Main metric display
        st.metric(
            label="Predicted Probability of Heterogeneous Frailty",
            value=f"{final_risk_prob * 100:.2f}%",
            delta=risk_cat,
            delta_color="inverse" if risk_cat == "High Risk" else "normal"
        )

        if badge_color == "success":
            st.success(f"**Clinical Recommendation:** {rec_text}")
        elif badge_color == "warning":
            st.warning(f"**Clinical Recommendation:** {rec_text}")
        else:
            st.error(f"**Clinical Recommendation:** {rec_text}")

        with st.expander("🔍 Base Learner Contributions", expanded=True):
            sub_col1, sub_col2, sub_col3 = st.columns(3)
            sub_col1.metric("Random Forest", f"{base_prob_dict['RF'] * 100:.1f}%")
            sub_col2.metric("CatBoost", f"{base_prob_dict['CatBoost'] * 100:.1f}%")
            sub_col3.metric("Extra Trees", f"{base_prob_dict['ET'] * 100:.1f}%")

            coef_df = pd.DataFrame({
                "Base Learner": ["Random Forest", "CatBoost", "Extra Trees"],
                "Meta-Learner Coefficient (Weight)": meta_learner.coef_[0]
            })
            st.dataframe(coef_df, use_container_width=True, hide_index=True)

    with col_shap:
        st.subheader("🔍 Local Feature Attribution (SHAP Waterfall)")
        st.caption(
            "SHAP feature attribution based on the primary tree-based estimator (Extra Trees). Red arrows increase the risk of frailty; blue arrows reduce it.")

        with st.spinner("Computing individual SHAP attribution..."):
            shap_values_single = explainer(input_df)

            # Handle binary multidimensional vs single-dimensional outputs
            if len(shap_values_single.shape) == 3:
                explanation = shap_values_single[0, :, 1]
            else:
                explanation = shap_values_single[0]

            fig, ax = plt.subplots(figsize=(8, 6))
            shap.plots.waterfall(explanation, max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

else:
    with col_pred:
        st.info(
            "👈 Enter or verify patient parameters in the sidebar, then click **'Calculate Risk & Explainability'**.")
    with col_shap:
        st.markdown("#### 📖 Global Feature Importance Reference")
        global_summary_img = os.path.join(MODEL_DIR, "shap_summary_et.png")
        if os.path.exists(global_summary_img):
            st.image(global_summary_img, caption="Global Feature Importance (SHAP Summary Plot - Extra Trees)",
                     use_container_width=True)