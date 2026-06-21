# =========================
# 0. 页面配置（必须最先）
# =========================
import streamlit as st
st.set_page_config(
    page_title="High Compensation Ratio Risk Prediction Model",
    layout="wide"
)

# =========================
# 1. 依赖
# =========================
import numpy as np
import pandas as pd
import joblib

# =========================
# 2. 加载逻辑回归模型
#    兼容两种保存格式：
#    A. 直接保存 sklearn LogisticRegression
#    B. 保存为 bundle: {"model": ..., "scaler": ..., "features": ...}
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
# 3. 工具函数
# =========================
def yes_no_to_int(value: str) -> int:
    return 1 if "Yes" in value else 0


def disability_grade_to_binary(grade: int) -> int:
    """
    后端二值化规则：
    1-4级 -> 1
    5-10级 -> 0
    """
    return 1 if 1 <= int(grade) <= 4 else 0


def prepare_model_input(x_original: pd.DataFrame):
    """按训练特征顺序排列；如 bundle 内含 scaler，则先做同样变换。"""
    x_ordered = x_original.reindex(columns=features)
    if scaler is not None:
        return scaler.transform(x_ordered)
    return x_ordered


def logistic_explanation(model, x_model, x_original: pd.DataFrame) -> pd.DataFrame:
    """返回逻辑回归在 log-odds 尺度上的逐变量贡献。"""
    if not hasattr(model, "coef_"):
        return pd.DataFrame()

    coef = np.asarray(model.coef_[0], dtype=float)

    if isinstance(x_model, pd.DataFrame):
        model_values = x_model.iloc[0].to_numpy(dtype=float)
    else:
        model_values = np.asarray(x_model, dtype=float).reshape(1, -1)[0]

    contributions = coef * model_values

    explanation_df = pd.DataFrame({
        "Feature": features,
        "Displayed input value": x_original.iloc[0].reindex(features).to_numpy(),
        "Model input value": model_values,
        "Coefficient": coef,
        "Log-odds contribution": contributions,
        "Odds ratio per 1-unit increase": np.exp(coef),
    })

    explanation_df["Direction"] = np.where(
        explanation_df["Log-odds contribution"] >= 0,
        "Increases high-risk probability",
        "Decreases high-risk probability"
    )

    return explanation_df.sort_values(
        "Log-odds contribution",
        key=lambda s: s.abs(),
        ascending=False
    )

# =========================
# 4. 页面标题
# =========================
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>"
    "High Compensation Ratio Risk Prediction Model"
    "</h1>",
    unsafe_allow_html=True
)

st.caption(f"Current model: {model_name}")

# =========================
# 5. 输入区域
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
        help="Backend encoding rule: grades 1–4 are encoded as 1; grades 5–10 are encoded as 0."
    )
    st.caption(
        "Backend encoding: disability severity grades 1–4 → 1; grades 5–10 → 0. "
        "Users should enter the actual grade only."
    )

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
# 6. 数值化（预测样本）
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

with st.expander("Check backend-coded model input"):
    coded_display = X_input.copy()
    coded_display.insert(
        0,
        "Disability severity grade entered by user",
        int(Disability_severity_grade_actual)
    )
    st.dataframe(coded_display, use_container_width=True, hide_index=True)

# =========================
# 7. 预测按钮
# =========================
st.markdown("---")
predict_btn = st.button("🔮 Predict Risk", use_container_width=True)

# =========================
# 8. 预测 + 结果展示 + 逻辑回归解释
# =========================
if predict_btn:

    # -------- 预测结果 --------
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

    st.caption("Classification threshold: predicted probability ≥ 0.500 is classified as High Risk.")

    # -------- 逻辑回归解释 --------
    st.markdown("---")
    st.subheader("🔍 Logistic Regression Explanation")

    explanation_df = logistic_explanation(model, X_model, X_input)

    if explanation_df.empty:
        st.warning("Coefficient-based explanation is unavailable for this model object.")
    else:
        intercept = float(np.asarray(model.intercept_).reshape(-1)[0])
        total_log_odds = intercept + explanation_df["Log-odds contribution"].sum()
        reconstructed_prob = 1 / (1 + np.exp(-np.clip(total_log_odds, -709, 709)))

        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            st.metric("Intercept", f"{intercept:.3f}")
        with col_e2:
            st.metric("Total log-odds", f"{total_log_odds:.3f}")
        with col_e3:
            st.metric("Probability from log-odds", f"{reconstructed_prob:.3f}")

        st.caption(
            "For logistic regression, each variable contributes coefficient × model input value "
            "to the linear predictor on the log-odds scale. Positive contributions push the prediction "
            "toward high risk; negative contributions push it toward low risk."
        )

        chart_df = explanation_df[["Feature", "Log-odds contribution"]].set_index("Feature")
        st.bar_chart(chart_df, use_container_width=True)

        st.dataframe(
            explanation_df[[
                "Feature",
                "Displayed input value",
                "Model input value",
                "Coefficient",
                "Log-odds contribution",
                "Odds ratio per 1-unit increase",
                "Direction",
            ]].round(4),
            use_container_width=True,
            hide_index=True
        )
