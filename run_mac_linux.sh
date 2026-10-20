#!/bin/bash
pip install -r requirements.txt
[ -f models/claim_risk_model.joblib ] || python train_model.py
streamlit run app.py
