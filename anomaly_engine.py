"""
anomaly_engine.py - the transparent (non-ML) layer.

1. Peer-group statistics: what is "normal" for a claim like this one?
   Peers = same vehicle class AND same injury severity (falls back to broader groups
   when a group has fewer than MIN_PEER_GROUP claims).
2. Ratios: claimed amount / peer median.
3. Rule flags: human-readable reasons an adjuster can read.

No fraud labels are used anywhere in this file, so it cannot leak the answer.
"""

import numpy as np
import pandas as pd

from schema import (
    INJURY_PHRASE, INJURY_ORD, LATE_FNOL_DAYS, MIN_PEER_GROUP,
    PEER_RATIO_FLAG, REPEAT_AT_FAULT,
)


def compute_peer_stats(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["inj"] = d["injury_severity_code"].map(INJURY_ORD)

    grp = d.groupby(["vehicle_class", "inj"])["gross_vehicle_damage_usd"].agg(["median", "size"])
    damage_ci = {k: float(r["median"]) for k, r in grp.iterrows() if r["size"] >= MIN_PEER_GROUP}
    damage_c = d.groupby("vehicle_class")["gross_vehicle_damage_usd"].median().astype(float).to_dict()

    injured = d[d["inj"] >= 1]
    medical_inj = injured.groupby("inj")["total_medical_cost_usd"].median().astype(float).to_dict()
    bi = injured[injured["bi_gross_claim_usd"] > 0]
    bi_inj = bi.groupby("inj")["bi_gross_claim_usd"].median().astype(float).to_dict()

    return {
        "damage_ci": damage_ci,
        "damage_c": damage_c,
        "damage_all": float(d["gross_vehicle_damage_usd"].median()),
        "medical_inj": medical_inj,
        "bi_inj": bi_inj,
        "n_claims": int(len(d)),
    }


def peer_damage(stats: dict, vclass: str, inj: int):
    """Median repair cost of comparable claims + a label describing the peer group."""
    nice = str(vclass).replace("_", " ")
    if (vclass, inj) in stats["damage_ci"]:
        return stats["damage_ci"][(vclass, inj)], f"{nice} claims with {INJURY_PHRASE[inj]}"
    if vclass in stats["damage_c"]:
        return stats["damage_c"][vclass], f"all {nice} claims"
    return stats["damage_all"], "all vehicle classes"


def peer_ratios(rec: dict, stats: dict) -> dict:
    """Claimed amount vs peer median for one claim (dict with raw fields)."""
    inj = int(INJURY_ORD[rec["injury_severity_code"]])
    dmg_peer, group = peer_damage(stats, rec["vehicle_class"], inj)
    damage = float(rec["gross_vehicle_damage_usd"])

    med_peer = stats["medical_inj"].get(inj)
    medical = float(rec.get("total_medical_cost_usd") or 0)
    bi_peer = stats["bi_inj"].get(inj)
    bi = float(rec.get("bi_gross_claim_usd") or 0)

    return {
        "inj": inj,
        "peer_group": group,
        "damage": damage,
        "damage_peer_median": dmg_peer,
        "damage_vs_peer": damage / dmg_peer if dmg_peer > 0 else 1.0,
        "medical": medical,
        "medical_peer_median": med_peer,
        "medical_vs_peer": medical / med_peer if (inj >= 1 and med_peer) else 1.0,
        "bi": bi,
        "bi_peer_median": bi_peer,
        "bi_vs_peer": bi / bi_peer if (inj >= 1 and bi_peer) else 1.0,
    }


def _times(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".") + " times"


def rule_flags(rec: dict, ratios: dict) -> list:
    """Plain-English warning signs. Each has a severity and a concrete 'check' step.
    severity: high / review / docs escalate the action to a human check; medium only informs."""
    flags = []
    inj = ratios["inj"]
    attorney = bool(rec.get("attorney_represented"))
    police = bool(rec.get("police_report_filed"))
    hit_run = bool(rec.get("hit_and_run_flag"))

    if attorney and inj == 0:
        flags.append({"code": "attorney_no_injury", "severity": "high",
                      "text": "A lawyer is involved, but no injury was reported",
                      "check": "Confirm with the claimant whether anyone was actually injured, and ask for medical records."})
    if (not police) and (inj >= 2 or hit_run):
        why = "a hit-and-run" if hit_run and inj < 2 else "a claim involving injuries"
        flags.append({"code": "no_police_serious", "severity": "docs",
                      "text": f"No police report was filed for {why}",
                      "check": "Ask for the police report, or the reason there isn't one."})
    if ratios["damage_vs_peer"] >= PEER_RATIO_FLAG:
        flags.append({"code": "damage_vs_peer", "severity": "review",
                      "text": f"Repair cost is {_times(ratios['damage_vs_peer'])} higher than similar claims "
                              f"(${ratios['damage']:,.0f} vs a typical ${ratios['damage_peer_median']:,.0f})",
                      "check": "Ask the repair shop for an itemised estimate and compare it with a second quote."})
    if inj >= 1 and ratios["medical_vs_peer"] >= PEER_RATIO_FLAG:
        flags.append({"code": "medical_vs_peer", "severity": "review",
                      "text": f"Medical bills are {_times(ratios['medical_vs_peer'])} higher than similar claims",
                      "check": "Request itemised medical bills and treatment records."})
    if inj >= 1 and ratios["bi_vs_peer"] >= PEER_RATIO_FLAG:
        flags.append({"code": "bi_vs_peer", "severity": "review",
                      "text": f"The injury payout requested is {_times(ratios['bi_vs_peer'])} higher than similar claims",
                      "check": "Review the injury payout request against the medical evidence."})
    delay = float(rec.get("fnol_delay_days") or 0)
    if delay >= LATE_FNOL_DAYS:
        flags.append({"code": "late_fnol", "severity": "medium",
                      "text": f"The claim was reported {delay:.0f} days after the accident",
                      "check": "Ask why the claim was reported late."})
    if float(rec.get("at_fault_accidents_3yr") or 0) >= REPEAT_AT_FAULT:
        flags.append({"code": "repeat_at_fault", "severity": "medium",
                      "text": "The driver has 2 or more at-fault accidents in the last 3 years",
                      "check": "Check the driver's claim history for repeat patterns."})
    if bool(rec.get("dui_flag")):
        flags.append({"code": "dui", "severity": "medium",
                      "text": "Driving under the influence is recorded on this claim",
                      "check": "Check the police and toxicology reports."})
    return flags
