# ClaimGuard AI - Auto Insurance Claim Risk Checker

Upload an accident claim, get a plain-English recommendation: **Approve**, **Review before paying**, or
**Investigate for possible fraud** - with the warning signs and what to check next.

Built on the XpertSystems INS-006 sample dataset (synthetic, CC-BY-NC-4.0).

## Run it
Windows: double-click `run_windows.bat`. Mac/Linux: `bash run_mac_linux.sh`.
(Installs packages, builds the model on first run, opens the app. The 3 data files are already in `data/`.)

Manual: `pip install -r requirements.txt` then `streamlit run app.py`.
The app builds the risk model automatically on first start (about a minute).

Optional AI summary: install Ollama and run `ollama pull qwen3:8b`. Without it a built-in summary is shown.

## Deploy
- **Docker:** `docker build -t claimguard .` then `docker run -p 8501:8501 claimguard`
  (the model is built inside the image; set `OLLAMA_URL` / `OLLAMA_MODEL` if an Ollama server is reachable).
- **Streamlit Community Cloud / any Python host:** push this folder (including `data/`) and run `streamlit run app.py`.
  The licence of the sample data is non-commercial - check it before any public or commercial deployment.

## How a claim is judged
1. `claim_parser.py` reads PDF / Excel / CSV / JSON. Missing required details -> no recommendation, never guessed.
2. `anomaly_engine.py` compares repair, medical and injury amounts with similar claims and writes plain warning signs.
3. `train_model.py` / `pipeline.py` score how closely the claim resembles claims investigators looked into
   (claim-time information only; `fraud_score` and all post-claim columns are excluded).
4. Rules can escalate a claim to review; they never lower a high pattern score.

## Honest notes
Only 31 of 5,000 claims are fraud-investigation cases, so metrics are indicative. All 17 "lawyer + no injury"
claims are investigation cases, and the system learns that pattern easily. The data is synthetic.

## Files
app.py (website) | pipeline.py | anomaly_engine.py | claim_parser.py | features.py | schema.py | train_model.py |
make_test_claims.py | smoke_test.py | .streamlit/config.toml (theme) | Dockerfile
