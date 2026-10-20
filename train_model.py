"""
train_model.py - trains the SIU-referral risk model and saves everything the app needs.

Run:  python train_model.py

Honest-evaluation choices (because only ~31 of 5,000 claims are SIU referrals):
  * repeated stratified 5-fold cross-validation (3 repeats) - never score on training rows
  * report PR-AUC and "recall at a fixed review budget", not accuracy
  * fraud_score and all post-claim columns are excluded (see schema.py)
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from anomaly_engine import compute_peer_stats
from features import build_features, compute_impute_values, load_master
from schema import FEATURE_COLS, REVIEW_PERCENTILE, SIU_PERCENTILE, TARGET

N_REPEATS = 3
REVIEW_BUDGET = 0.03  # "if we can only look at the top 3% of claims..."

CANDIDATES = {
    "Logistic Regression": lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", C=0.3, max_iter=3000),
    ),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample",
        random_state=0, n_jobs=-1,
    ),
}


def recall_precision_at_budget(y, score, budget):
    k = max(1, int(round(budget * len(y))))
    top = np.argsort(-score)[:k]
    hits = int(y[top].sum())
    return hits / max(1, int(y.sum())), hits / k


def main():
    master = load_master("data")
    y = master[TARGET].astype(int).values
    stats = compute_peer_stats(master)
    impute = compute_impute_values(master)
    X = build_features(master, stats, impute)

    print(f"Claims: {len(y)} | SIU referrals: {y.sum()} ({y.mean()*100:.2f}%) | features: {X.shape[1]}")

    results, oof_by_model = {}, {}
    for name, make in CANDIDATES.items():
        per_rep, oofs = [], []
        for rep in range(N_REPEATS):
            skf = StratifiedKFold(5, shuffle=True, random_state=rep)
            oof = cross_val_predict(make(), X, y, cv=skf, method="predict_proba")[:, 1]
            rec, prec = recall_precision_at_budget(y, oof, REVIEW_BUDGET)
            per_rep.append({
                "roc_auc": roc_auc_score(y, oof),
                "pr_auc": average_precision_score(y, oof),
                "recall_at_budget": rec, "precision_at_budget": prec,
            })
            oofs.append(oof)
        df_rep = pd.DataFrame(per_rep)
        results[name] = {
            "mean": df_rep.mean().round(4).to_dict(),
            "std": df_rep.std().round(4).to_dict(),
        }
        oof_by_model[name] = np.mean(oofs, axis=0)
        m = results[name]["mean"]; s = results[name]["std"]
        print(f"{name:20} ROC-AUC {m['roc_auc']:.3f}±{s['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f}±{s['pr_auc']:.3f} "
              f"| top-{REVIEW_BUDGET:.0%}: recall {m['recall_at_budget']:.2f}, precision {m['precision_at_budget']:.2f}")

    best = max(results, key=lambda k: results[k]["mean"]["pr_auc"])
    print(f"\nSelected model (best PR-AUC): {best}")

    oof = oof_by_model[best]
    # Platt scaling on out-of-fold scores: turns a raw score into "how often did claims
    # with this score actually get referred?"
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(logit(oof).reshape(-1, 1), y)

    final = CANDIDATES[best]().fit(X, y)

    # global importance for the report
    if hasattr(final, "feature_importances_"):
        imp = final.feature_importances_
    else:
        imp = np.abs(final[-1].coef_[0])
        imp = imp / imp.sum()
    top_imp = sorted(zip(FEATURE_COLS, imp.round(4).tolist()), key=lambda x: -x[1])[:10]

    # transparency about the generator: which share of positives is one simple rule?
    attorney_noinj = (master["attorney_represented"] & (master["injury_severity_code"] == "O_no_injury"))
    rule_stats = {
        "attorney_no_injury_claims": int(attorney_noinj.sum()),
        "of_which_siu": int(master.loc[attorney_noinj, TARGET].sum()),
        "siu_total": int(y.sum()),
    }
    leak = {
        "fraud_score_threshold_match_pct": round(float(((master["fraud_score"] > 65) == master[TARGET]).mean() * 100), 2),
        "fraud_suspected_flag_positives": int(master["fraud_suspected_flag"].sum()),
    }

    bundle = {
        "model": final,
        "model_name": best,
        "feature_cols": FEATURE_COLS,
        "peer_stats": stats,
        "impute": impute,
        "feature_medians": X.median().to_dict(),
        "calibrator": calibrator,
        "oof": pd.DataFrame({"claim_number": master["claim_number"], "oof_score": oof}),
        "oof_sorted": np.sort(oof),
        "thresholds": {"siu_pct": SIU_PERCENTILE, "review_pct": REVIEW_PERCENTILE},
    }
    os.makedirs("models", exist_ok=True)
    joblib.dump(bundle, "models/claim_risk_model.joblib")

    report = {
        "n_claims": int(len(y)), "n_positive": int(y.sum()), "base_rate_pct": round(float(y.mean() * 100), 3),
        "selected_model": best, "review_budget": REVIEW_BUDGET, "cv": f"{N_REPEATS} x 5-fold stratified",
        "candidates": results, "top_features": top_imp, "rule_coverage": rule_stats, "leakage_check": leak,
        "excluded_columns": ["fraud_score", "fraud_suspected_flag", "claim_status", "litigation_flag",
                             "settlement_days", "alae_usd", "case_reserve_usd", "paid_to_date_usd",
                             "total_gross_incurred_loss_usd", "loss_development_factor_*"],
    }
    with open("models/metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved models/claim_risk_model.joblib and models/metrics.json")


if __name__ == "__main__":
    main()
