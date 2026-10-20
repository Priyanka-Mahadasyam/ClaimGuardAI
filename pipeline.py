"""
pipeline.py - takes ONE parsed claim (dict) and returns a full, explainable verdict.
No Streamlit in here, so it can be tested on its own.

The result is written in plain English first (decision, reason, warning signs, next steps);
the technical numbers (percentile, probability, model name) live under result["technical"].
"""

import os
import re

import joblib
import numpy as np
import pandas as pd

from anomaly_engine import peer_ratios, rule_flags
from features import build_features
from schema import (
    ACT_APPROVE, ACT_INVESTIGATE, ACT_REVIEW, ACTION_UI, BOOL_DEFAULT_FALSE, DISPLAY,
    FEATURE_COLS, INJURY_ORD, PLAIN_FEATURE, RAW_BOOL, RAW_NUMERIC, REQUIRED,
    REVIEW_PERCENTILE, SIU_PERCENTILE,
)

MODEL_PATH = os.path.join("models", "claim_risk_model.joblib")


def load_bundle(path: str = MODEL_PATH) -> dict:
    return joblib.load(path)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _raw_score(bundle, X: pd.DataFrame) -> np.ndarray:
    return bundle["model"].predict_proba(X[bundle["feature_cols"]])[:, 1]


def score_to_percentile(bundle, raw: float) -> float:
    ref = bundle["oof_sorted"]
    return float(np.searchsorted(ref, raw, side="right") / len(ref) * 100)


def score_to_probability(bundle, raw: float) -> float:
    return float(bundle["calibrator"].predict_proba(_logit(np.array([[raw]])))[0, 1])


def prepare_claim(claim: dict, bundle: dict):
    """Validate + fill optional gaps. Returns (record, filled_fields, errors)."""
    rec = dict(claim)
    errors = [DISPLAY[k] for k in REQUIRED if rec.get(k) is None or rec.get(k) == ""]
    if errors:
        return rec, [], errors

    filled = []
    for c in RAW_NUMERIC:
        if rec.get(c) is None or (isinstance(rec.get(c), float) and np.isnan(rec[c])):
            if c == "ubi_score" and not rec.get("telematics_enrolled"):
                rec[c] = bundle["impute"][c]  # not enrolled: value is irrelevant, no warning
                continue
            if c in ("total_medical_cost_usd", "bi_gross_claim_usd") and INJURY_ORD[rec["injury_severity_code"]] == 0:
                rec[c] = 0.0  # no injury -> no medical / injury claim, nothing assumed
                continue
            rec[c] = bundle["impute"][c]
            filled.append(DISPLAY.get(c, c))
    for c in RAW_BOOL:
        if rec.get(c) is None:
            rec[c] = False
            if c in BOOL_DEFAULT_FALSE:
                filled.append(DISPLAY.get(c, c) + " (assumed No)")
    return rec, filled, []


# ------------------------------------------------------------------ explanations
def _local_drivers(bundle, X_row: pd.DataFrame, base_raw: float, top_n: int = 6, min_pp: float = 0.1):
    """Which inputs pushed the pattern score up? Replace each input with the portfolio median
    and see how much the estimated probability drops (percentage points). Tiny effects are hidden."""
    medians = bundle["feature_medians"]
    base_prob = score_to_probability(bundle, base_raw)
    out = []
    for col in FEATURE_COLS:
        alt = X_row.copy()
        alt[col] = medians[col]
        alt_prob = score_to_probability(bundle, float(_raw_score(bundle, alt)[0]))
        delta_pp = (base_prob - alt_prob) * 100
        if delta_pp >= min_pp:
            out.append((col, delta_pp, float(X_row[col].iloc[0])))
    out.sort(key=lambda o: -o[1])
    return [{"key": c, "feature": DISPLAY.get(c, c), "impact": d, "value": v} for c, d, v in out[:top_n]]


def _plain_reasons(drivers: list, max_items: int = 4) -> list:
    """Model drivers -> short plain phrases with a strength label."""
    out = []
    for d in drivers[:max_items]:
        phrase = PLAIN_FEATURE.get(d["key"], d["feature"].lower())
        strength = "Strong" if d["impact"] >= 30 else "Moderate" if d["impact"] >= 5 else "Slight"
        out.append({"text": phrase[0].upper() + phrase[1:], "strength": strength})
    return out


def ml_level(percentile: float) -> str:
    if percentile >= SIU_PERCENTILE:
        return "VERY HIGH"
    if percentile >= REVIEW_PERCENTILE:
        return "HIGH"
    if percentile >= 50:
        return "MODERATE"
    return "LOW"


def decide(percentile: float, flags: list):
    """Recommended action + a plain-English reason.
    Rules can escalate, they never lower a high pattern score.
    high = strong red flag, review = price outlier vs similar claims, docs = missing paperwork."""
    if percentile >= SIU_PERCENTILE:
        return ACT_INVESTIGATE, "Its overall pattern closely matches claims that investigators looked into in the past"
    high = [f["text"] for f in flags if f["severity"] == "high"]
    if high:
        return ACT_REVIEW, "Warning sign: " + high[0][0].lower() + high[0][1:]
    if percentile >= REVIEW_PERCENTILE:
        return ACT_REVIEW, "Its overall pattern is unusual compared with past claims"
    triggers = [f["text"] for f in flags if f["severity"] in ("review", "docs")]
    if triggers:
        return ACT_REVIEW, "Needs checking: " + "; ".join(t[0].lower() + t[1:] for t in triggers)
    return ACT_APPROVE, "No warning signs were found and the overall pattern looks normal"


def _next_steps(action: str, flags: list) -> list:
    if action == ACT_APPROVE:
        return ["No extra checks needed. Process the claim as normal."]
    steps = []
    if action == ACT_INVESTIGATE:
        steps.append("Send the full claim file to the fraud investigation team.")
    steps += [f["check"] for f in flags if f.get("check")]
    if action == ACT_REVIEW and not flags:
        steps.append("Ask for supporting documents (photos, repair estimate, medical records) to confirm the details.")
    seen, out = set(), []
    for s in steps:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _status(ratio: float):
    if ratio >= 3:
        return "Much higher", "bad"
    if ratio >= 1.5:
        return "Higher than usual", "warn"
    if ratio <= 0.6:
        return "Lower than usual", "info"
    return "Normal", "ok"


def _peer_rows(ratios: dict) -> list:
    rows = []

    def add(label, claim, typical, ratio):
        status, level = _status(ratio)
        rows.append({"check": label, "claim": f"${claim:,.0f}", "typical": f"${typical:,.0f}",
                     "ratio": ratio, "status": status, "level": level})

    add("Repair cost", ratios["damage"], ratios["damage_peer_median"], ratios["damage_vs_peer"])
    if ratios["inj"] >= 1:
        if ratios["medical_peer_median"]:
            add("Medical bills", ratios["medical"], ratios["medical_peer_median"], ratios["medical_vs_peer"])
        if ratios["bi_peer_median"]:
            add("Injury payout requested", ratios["bi"], ratios["bi_peer_median"], ratios["bi_vs_peer"])
    return rows


# ------------------------------------------------------------------ main entry point
def score_claim(claim: dict, bundle: dict, raw_override: float = None) -> dict:
    """raw_override: use a stored out-of-sample score (for historical claims) instead of re-scoring."""
    rec, filled, errors = prepare_claim(claim, bundle)
    if errors:
        return {"ok": False, "missing_required": errors}

    df = pd.DataFrame([rec])
    X = build_features(df, bundle["peer_stats"], bundle["impute"])
    live_raw = float(_raw_score(bundle, X)[0])
    raw = live_raw if raw_override is None else float(raw_override)
    pct = score_to_percentile(bundle, raw)
    prob = score_to_probability(bundle, raw)

    ratios = peer_ratios(rec, bundle["peer_stats"])
    flags = rule_flags(rec, ratios)
    action, reason = decide(pct, flags)
    drivers = _local_drivers(bundle, X, live_raw)
    ui = ACTION_UI[action]

    return {
        "ok": True,
        "claim": rec,
        "action": action,
        "action_title": ui["title"],
        "level": ui["level"],
        "action_reason": reason,
        "warning_signs": flags,
        "reasons": _plain_reasons(drivers),
        "next_steps": _next_steps(action, flags),
        "peer_rows": _peer_rows(ratios),
        "peer_group": ratios["peer_group"],
        "filled_fields": filled,
        "technical": {
            "est_probability": prob, "raw_score": raw,
            "ml_level": ml_level(pct), "model_name": bundle["model_name"], "drivers": drivers,
        },
    }


# ------------------------------------------------------------------ AI summary (Ollama)
def build_prompt(result: dict) -> str:
    signs = "\n".join(f"- {f['text']}" for f in result["warning_signs"]) or "- none"
    peers = "\n".join(f"- {p['check']}: {p['claim']} claimed vs {p['typical']} typical for similar claims ({p['status']})"
                      for p in result["peer_rows"])
    steps = "\n".join(f"- {s}" for s in result["next_steps"])
    return f"""You are writing a short note for an insurance claims adjuster who is NOT a data expert.

RULES
- Use simple everyday words. Do NOT use technical terms such as ML, model, percentile, SIU, score or algorithm.
- Use ONLY the facts below. Do not invent numbers. Do not mention anything that is not listed.
- Never say the claim IS fraud. Say what looks unusual and what to check.
- If there are no warning signs, say plainly that nothing unusual was found.

Decision: {result['action_title']}
Why: {result['action_reason']}
Warning signs:
{signs}

Comparison with similar claims:
{peers}

Suggested checks:
{steps}

Write 3 to 4 short sentences: first the decision and why, then the main evidence, then what to check next."""


def fallback_explanation(result: dict) -> str:
    """Used when the local LLM is not running - built only from computed facts, in plain words."""
    parts = [f"**Decision: {result['action_title']}.** {result['action_reason']}."]
    reason_has_signs = result["action_reason"].lower().startswith(("needs checking", "warning sign"))
    if result["warning_signs"] and not reason_has_signs:
        parts.append("Warning signs: " + "; ".join(f["text"] for f in result["warning_signs"]) + ".")
    parts.append("What to do next: " + " ".join(result["next_steps"]))
    return " ".join(parts)


def ask_ollama(prompt: str, model: str = None, url: str = None, timeout: int = 120) -> str:
    """Local Qwen via Ollama. Override with env vars OLLAMA_URL / OLLAMA_MODEL when deploying."""
    import requests
    model = model or os.environ.get("OLLAMA_MODEL", "qwen3:8b")
    url = url or os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    resp = requests.post(url, json={"model": model, "prompt": prompt, "stream": False, "think": False,
                                    "options": {"temperature": 0.2}}, timeout=timeout)
    resp.raise_for_status()
    text = resp.json().get("response", "")
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()  # never show reasoning tags
