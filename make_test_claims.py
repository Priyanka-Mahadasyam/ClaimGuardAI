"""
make_test_claims.py - builds sample claim documents in sample_claims/ for testing the app.

Run:  python make_test_claims.py
Needs: reportlab  (pip install reportlab)  - only for creating test PDFs, not for the app.
"""

import json
import os

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from anomaly_engine import compute_peer_stats, peer_ratios
from features import load_master
from schema import INJURY_ORD

OUT = "sample_claims"
os.makedirs(OUT, exist_ok=True)

INJ_TEXT = {
    "O_no_injury": "O - No injury", "C_possible_injury": "C - Possible injury",
    "B_non_incapacitating": "B - Non-incapacitating injury",
    "A_incapacitating": "A - Incapacitating injury", "K_fatal": "K - Fatal",
}


def yn(v):
    return "Yes" if bool(v) else "No"


def usd(v):
    return f"${float(v):,.2f}"


def form_sections(c: dict):
    """(section title, [(label, value-or-None)]) - None values are left OUT of the form."""
    return [
        ("CLAIM", [("Claim Number", c.get("claim_number")), ("Policy ID", c.get("policy_id"))]),
        ("VEHICLE", [
            ("Vehicle Class", str(c["vehicle_class"]).replace("_", " ").title() if c.get("vehicle_class") else None),
            ("Vehicle Make", c.get("vehicle_make")), ("Vehicle Model", c.get("vehicle_model")),
            ("Vehicle Age (years)", c.get("vehicle_age_years")),
            ("Actual Cash Value (USD)", usd(c["actual_cash_value_usd"]) if c.get("actual_cash_value_usd") is not None else None),
        ]),
        ("ACCIDENT", [
            ("Accident Type", str(c.get("accident_type", "")).replace("_", " ").title() or None),
            ("Vehicles Involved", c.get("vehicles_involved_count")),
            ("Hit and Run", yn(c.get("hit_and_run_flag"))),
            ("Police Report Filed", yn(c.get("police_report_filed"))),
            ("Reporting Delay (days)", c.get("fnol_delay_days")),
            ("Insured At-Fault %", c.get("at_fault_pct_insured")),
        ]),
        ("INJURY AND CLAIM AMOUNTS", [
            ("Injury Severity", INJ_TEXT.get(c.get("injury_severity_code"))),
            ("Claimant Injuries", c.get("injuries_count_claimant")),
            ("Total Medical Cost (USD)", usd(c["total_medical_cost_usd"])),
            ("Bodily Injury Claim (USD)", usd(c["bi_gross_claim_usd"])),
            ("Attorney Represented", yn(c.get("attorney_represented"))),
            ("Gross Vehicle Damage (USD)", usd(c["gross_vehicle_damage_usd"]) if c.get("gross_vehicle_damage_usd") is not None else None),
        ]),
        ("DRIVER AND POLICY", [
            ("Driver Age", c.get("age")), ("Years Licensed", c.get("years_licensed")),
            ("MVR Points", c.get("mvr_points")), ("MVR Violations (3 yr)", c.get("mvr_violations_3yr")),
            ("At-Fault Accidents (3 yr)", c.get("at_fault_accidents_3yr")),
            ("Credit Score", c.get("credit_score")), ("Insurance Score", c.get("insurance_score")),
            ("Annual Mileage", c.get("annual_mileage")), ("DUI Flag", yn(c.get("dui_flag"))),
            ("Annual Premium (USD)", usd(c["annual_premium_usd"])),
            ("Telematics Enrolled", yn(c.get("telematics_enrolled"))),
            ("UBI Score", c.get("ubi_score") if pd.notna(c.get("ubi_score")) else None),
        ]),
    ]


def write_pdf(path, claim, note):
    cv = canvas.Canvas(path, pagesize=A4)
    w, h = A4
    y = h - 55
    cv.setFont("Helvetica-Bold", 16)
    cv.drawString(50, y, "AUTO INSURANCE CLAIM SUBMISSION FORM")
    y -= 16
    cv.setFont("Helvetica", 8)
    cv.drawString(50, y, f"Synthetic test document - {note}")
    y -= 22
    for title, rows in form_sections(claim):
        cv.setFont("Helvetica-Bold", 10)
        cv.drawString(50, y, title)
        y -= 4
        cv.line(50, y, w - 50, y)
        y -= 14
        cv.setFont("Helvetica", 10)
        for label, val in rows:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            if isinstance(val, float) and val == int(val):
                val = int(val)
            cv.drawString(60, y, f"{label}: {val}")
            y -= 14
        y -= 8
    cv.save()


def main():
    m = load_master("data").copy()
    stats = compute_peer_stats(m)
    m["inj"] = m["injury_severity_code"].map(INJURY_ORD)
    m["dmg_vs"] = [peer_ratios(r.to_dict(), stats)["damage_vs_peer"] for _, r in m.iterrows()]
    siu = m["siu_referral_flag"]

    # 1. clean, typical claim
    clean = m[(m.inj == 0) & (~m.attorney_represented) & m.police_report_filed & (~siu)
              & (m.dmg_vs.between(0.9, 1.1)) & (m.vehicle_class == "sedan")].iloc[0].to_dict()
    # 2. real SIU referral from the dataset (attorney + no injury)
    cluster = m[siu & (m.inj == 0)].iloc[0].to_dict()
    # 3. injury claim, no police report, late reporting (hand-modified real claim)
    docs = m[(m.inj == 2) & (~siu) & (m.vehicle_class == "suv_crossover")].iloc[3].to_dict()
    docs.update(police_report_filed=False, fnol_delay_days=9)
    # 4. large but legitimate fatal claim (hard negative: big numbers are not fraud by themselves)
    fatal = m[(m.inj == 4) & (~siu) & m.attorney_represented].sort_values("bi_gross_claim_usd").iloc[-5].to_dict()
    # 5. repair cost 4x the peer median on an otherwise normal claim
    inflated = dict(clean)
    inflated["gross_vehicle_damage_usd"] = round(clean["gross_vehicle_damage_usd"] * 4.0, 2)
    inflated["claim_number"] = "CLM-TEST-INFLATED"
    # 6. incomplete form (required fields removed)
    incomplete = dict(clean)
    incomplete["injury_severity_code"] = None
    incomplete["gross_vehicle_damage_usd"] = None
    incomplete["claim_number"] = "CLM-TEST-INCOMPLETE"

    cases = [
        ("claim_1_clean_typical.pdf", clean, "typical fender-bender, expect fast-track"),
        ("claim_2_attorney_no_injury.pdf", cluster, "real SIU referral from the dataset (in training data)"),
        ("claim_3_missing_police_report.pdf", docs, "injury claim, no police report, late reporting"),
        ("claim_4_large_legit_fatal.pdf", fatal, "large fatal claim - big is not automatically fraud"),
        ("claim_5_repair_cost_outlier.pdf", inflated, "repair cost 4x peer median"),
        ("claim_6_incomplete_form.pdf", incomplete, "required fields missing - app must stop"),
    ]
    for fname, claim, note in cases:
        write_pdf(os.path.join(OUT, fname), claim, note)
        print("wrote", fname)

    # tabular samples use the dataset's own column names
    cols = ["claim_number", "policy_id", "vehicle_class", "vehicle_age_years", "actual_cash_value_usd",
            "vehicles_involved_count", "hit_and_run_flag", "police_report_filed", "fnol_delay_days",
            "at_fault_pct_insured", "injury_severity_code", "injuries_count_claimant", "total_medical_cost_usd",
            "bi_gross_claim_usd", "attorney_represented", "gross_vehicle_damage_usd", "age", "years_licensed",
            "mvr_points", "mvr_violations_3yr", "at_fault_accidents_3yr", "credit_score", "insurance_score",
            "annual_mileage", "dui_flag", "annual_premium_usd", "telematics_enrolled"]
    row = pd.DataFrame([docs])[cols]
    row.to_csv(os.path.join(OUT, "sample_claim.csv"), index=False)
    with open(os.path.join(OUT, "sample_claim.json"), "w") as f:
        json.dump(json.loads(row.to_json(orient="records")), f, indent=2)
    print("wrote sample_claim.csv / sample_claim.json")


if __name__ == "__main__":
    main()
