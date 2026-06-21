# High Compensation Ratio Risk Prediction Model

This Streamlit app uses a logistic regression model to predict high compensation ratio risk.

## Files

- `app.py`: Streamlit app
- `model.pkl`: trained logistic regression model bundle
- `requirements.txt`: Python dependencies

## Local run

```bash
streamlit run app.py
```

or

```bash
python -m streamlit run app.py
```

## Disability severity encoding

Users enter the actual disability severity grade from 1 to 10.

Backend encoding rule:

- Grades 1–4 are encoded as 1
- Grades 5–10 are encoded as 0

## Logistic regression explanation

The app shows:

- predicted high-risk probability
- risk class based on threshold 0.5
- logistic regression probability curve
- feature contribution plot on the log-odds scale
- optional coefficient table
