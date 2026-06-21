# High Compensation Ratio Risk Prediction Model

This Streamlit app uses a logistic regression model to predict high compensation ratio risk.

## Files

- `app.py`: Streamlit application.
- `model.pkl`: Logistic regression model bundle. The app supports a bundle with `model`, `scaler`, and `features`.
- `requirements.txt`: Python dependencies for Streamlit deployment.

## Key input rule

Users enter the actual disability severity grade from 1 to 10. The app converts this value in the backend:

- Grades 1–4 are encoded as `1`.
- Grades 5–10 are encoded as `0`.

This avoids asking users to manually choose the binary coding.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
