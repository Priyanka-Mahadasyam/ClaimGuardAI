"""
features.py - loads the 3 dataset files and builds model features.
The SAME build_features() is used for training and for a newly uploaded claim, so
the model never sees a feature at upload time that it did not see in training.
"""

import os

import numpy as np
import pandas as pd

from anomaly_engine import peer_ratios
from schema import (
    DERIVED, FEATURE_COLS, INJURY_ORD, LATE_FNOL_DAYS, RAW_BOOL, RAW_NUMERIC,
)

DRIVER_COLS = [
    "age", "years_licensed", "mvr_points", "mvr_violations_3yr",
    "at_fault_accidents_3yr", "credit_score", "insurance_score",
    "annual_mileage", "dui_flag",
]


def load_master(data_dir: str = "data") -> pd.DataFrame:
    """auto_accident_records + driver_profiles + telematics_summary joined on policy_id."""
    acc = pd.read_csv(os.path.join(data_dir, "auto_accident_records.csv"))
    drv = pd.read_csv(os.path.join(data_dir, "driver_profiles.csv"))
    tel = pd.read_csv(os.path.join(data_dir, "telematics_summary.csv"))

    # Driver + telematics columns come from their own files (the accident file only
    # carries empty placeholders for telematics).
    drop = [c for c in DRIVER_COLS + ["telematics_enrolled", "ubi_score"] if c in acc.columns]
    acc = acc.drop(columns=drop)
    acc = acc.merge(drv[["policy_id"] + DRIVER_COLS], on="policy_id", how="left")
    acc = acc.merge(tel[["policy_id", "telematics_enrolled", "ubi_score"]], on="policy_id", how="left")
    return acc.copy()  # de-fragment after the merges


def compute_impute_values(df: pd.DataFrame) -> dict:
    """Portfolio medians used to fill optional numeric fields that a document leaves out."""
    imp = {c: float(pd.to_numeric(df[c], errors="coerce").median()) for c in RAW_NUMERIC}
    enrolled = df[df["telematics_enrolled"].fillna(False).astype(bool)]
    imp["ubi_score"] = float(enrolled["ubi_score"].median())
    return imp


def _derive_row(r: pd.Series, stats: dict) -> dict:
    ratios = peer_ratios(r.to_dict(), stats)
    inj = ratios["inj"]
    acv = float(r["actual_cash_value_usd"])
    attorney = bool(r["attorney_represented"])
    police = bool(r["police_report_filed"])
    return {
        "injury_ord": inj,
        "damage_ratio": float(r["gross_vehicle_damage_usd"]) / acv if acv > 0 else 1.0,
        "damage_vs_peer": ratios["damage_vs_peer"],
        "medical_vs_peer": ratios["medical_vs_peer"],
        "bi_vs_peer": ratios["bi_vs_peer"],
        "attorney_no_injury": int(attorney and inj == 0),
        "late_fnol": int(float(r["fnol_delay_days"]) >= LATE_FNOL_DAYS),
        "no_police_serious": int((not police) and (inj >= 2 or bool(r["hit_and_run_flag"]))),
    }


def build_features(df: pd.DataFrame, stats: dict, impute: dict) -> pd.DataFrame:
    """Raw claim rows -> model matrix (FEATURE_COLS, no NaN)."""
    raw = df.copy()
    for c in RAW_BOOL:
        raw[c] = raw[c].fillna(False).astype(bool)
    for c in RAW_NUMERIC:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(impute[c])

    derived = pd.DataFrame([_derive_row(r, stats) for _, r in raw.iterrows()], index=raw.index)

    X = pd.concat([raw[RAW_NUMERIC], raw[RAW_BOOL].astype(int), derived[DERIVED]], axis=1)
    return X[FEATURE_COLS].astype(float)
