@echo off
echo === Installing packages ===
pip install -r requirements.txt
if not exist models\claim_risk_model.joblib (
  echo === Training model - about 1 minute ===
  python train_model.py
)
echo === Starting app ===
streamlit run app.py
pause
