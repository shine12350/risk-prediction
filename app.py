# =========================
# 0. Page config: must be first Streamlit command
# =========================
import streamlit as st

st.set_page_config(
    page_title="High Compensation Ratio Risk Prediction Model",
    layout="wide"
)

# =========================
# 1. Dependencies
# =========================
import textwrap

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =========================
# 2. Load logistic regression model
#    Compatible with either:
#    A. a directly saved sklearn LogisticRegression object
#    B. a saved bundle: {"model": ..., "scaler": ..., "features": ...}
# =========================
MODEL_PATH = "model.pkl"

bundle = joblib.load(MODEL_PATH)

if isinstance(bundle, dict):
    model = bundle["model"]
    scaler = bundle.get("scaler", None)
    features = list(bundle.get("features", []))
    model_name = bundle.get("model_name", "Logistic Regression")
else:
    model = bundle
    scaler = None
    features = list(getattr(model, "feature_names_in_", []))
    model_name = "Logistic Regression"

if not features:
    st.error("Model features were not found. Please save feature names in the model bundle.")
    st.stop()

# =========================
# 3. Helper functions
# =========================
def yes_no_to_int(value: str) -> int:
    return 1 if "Yes" in value else 0


def disability_grade_to_binary(grade: int) -> int:
    """
    Backend encoding rule:
    grades 1-4 -> 1
    grades 5-10 -> 0
    """
    return 1 if 1 <= int(grade) <= 4 else 0


def prepare_model_input(x_original: pd.DataFrame):
    """Order features exactly as in training; apply scaler if included in the bundle."""
    x_ordered = x_original.reindex(columns=features)
    if scaler is not None:
        return scaler.transform(x_ordered)
    return x_ordered


def logistic_contributions(model, x_model) -> pd.DataFrame:
    """Calculate feature contributions on the log-odds scale: coefficient * model input value."""
    if not hasattr(model, "coef_"):
        return pd.DataFrame()

    coef = np.asarray(model.coef_[0], dtype=float)

    if isinstance(x_model, pd.DataFrame):
        model_values = x_model.iloc[0].to_numpy(dtype=float)
    else:
        model_values = np.asarray(x_model, dtype=float).reshape(1, -1)[0]

    contributions = coef * model_values

    contribution_df = pd.DataFrame({
        "Feature": features,
        "Contribution": contributions,
    })

    contribution_df["Display feature"] = contribution_df["Feature"].map(pretty_feature_name)

    return contribution_df.sort_values(
        "Contribution",
        key=lambda s: s.abs(),
        ascending=True
    )


def pretty_feature_name(feature: str) -> str:
    mapping = {
        "Gender": "Gender",
        "Age": "Age",
        "Disability_severity_grade": "Disability severity grade",
        "Inappropriate_surgical_procedure": "Inappropriate surgical procedure",
        "Inadequate_medical_records": "Inadequate medical records",
        "Lack_of_informed_consent": "Lack of informed consent",
        "Treatment_delay": "Treatment delay",
        "Inadequate_preoperative_preparation": "Inadequate preoperative preparation",
    }
    return mapping.get(feature, feature.replace("_", " "))


def plot_contributions(contribution_df: pd.DataFrame):
    """Horizontal contribution plot with readable variable names."""
    plot_df = contribution_df.copy()
    plot_df["Wrapped feature"] = plot_df["Display feature"].apply(
        lambda x: "\n".join(textwrap.wrap(x, width=28))
    )

    fig_height = max(3.6, 0.45 * len(plot_df) + 1.4)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    ax.barh(plot_df["Wrapped feature"], plot_df["Contribution"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Contribution to predicted high-risk probability on the log-odds scale")
    ax.set_ylabel("")
    ax.set_title("Variable Contribution")

    for i, value in enumerate(plot_df["Contribution"]):
        offset = 0.03 if value >= 0 else -0.03
        ha = "left" if value >= 0 else "right"
        ax.text(value + offset, i, f"{value:.2f}", va="center", ha=ha, fontsize=9)

    fig.tight_layout()
    return fig

# =========================
# 4. Page title
# =========================
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>"
    "High Compensation Ratio Risk Prediction Model"
    "</h1>",
    unsafe_allow_html=True
)


# =========================
# 5. Input area
# =========================
col1, col2 = st.columns(2)

with col1:
    Gender = st.selectbox("Gender", ["Male (1)", "Female (0)"])
    Age = st.number_input("Age", min_value=0, max_value=120, value=40, step=1)

    Disability_severity_grade_actual = st.number_input(
        "Disability severity grade (actual grade: 1–10)",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Enter the actual grade only. The app handles the internal model encoding automatically."
    )
    st.caption("Enter the actual 1–10 grade only; internal model encoding is handled automatically.")

    Inappropriate_surgical_procedure = st.selectbox(
        "Inappropriate surgical procedure", ["Yes (1)", "No (0)"]
    )

with col2:
    Inadequate_medical_records = st.selectbox(
        "Inadequate medical records", ["Yes (1)", "No (0)"]
    )
    Lack_of_informed_consent = st.selectbox(
        "Lack of informed consent", ["Yes (1)", "No (0)"]
    )
    Treatment_delay = st.selectbox(
        "Treatment delay", ["Yes (1)", "No (0)"]
    )
    Inadequate_preoperative_preparation = st.selectbox(
        "Inadequate preoperative preparation", ["Yes (1)", "No (0)"]
    )

# =========================
# 6. Build prediction sample
# =========================
disability_binary = disability_grade_to_binary(Disability_severity_grade_actual)

X_input = pd.DataFrame([{
    "Gender": 1 if "Male" in Gender else 0,
    "Age": int(Age),
    "Disability_severity_grade": disability_binary,
    "Inappropriate_surgical_procedure": yes_no_to_int(Inappropriate_surgical_procedure),
    "Inadequate_medical_records": yes_no_to_int(Inadequate_medical_records),
    "Lack_of_informed_consent": yes_no_to_int(Lack_of_informed_consent),
    "Treatment_delay": yes_no_to_int(Treatment_delay),
    "Inadequate_preoperative_preparation": yes_no_to_int(Inadequate_preoperative_preparation),
}])

X_input = X_input.reindex(columns=features)
X_model = prepare_model_input(X_input)

# =========================
# 7. Prediction button
# =========================
st.markdown("---")
predict_btn = st.button("🔮 Predict Risk", width="stretch")

# =========================
# 8. Prediction result only
# =========================
if predict_btn:
    prob = float(model.predict_proba(X_model)[0][1])
    pred = 1 if prob >= 0.5 else 0

    st.subheader("📊 Prediction Result")

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.metric(
            label="Predicted High-risk Probability",
            value=f"{prob:.3f}"
        )

    with col_r2:
        st.metric(
            label="Predicted Risk Level",
            value="High Risk" if pred == 1 else "Low Risk"
        )

    st.markdown("#### High-risk probability visualization")
    st.write(f"High-risk probability: **{prob:.1%}**")
    st.progress(int(round(prob * 100)))

    if pred == 1:
        st.error("The model classifies this case as High Risk.")
    else:
        st.success("The model classifies this case as Low Risk.")

