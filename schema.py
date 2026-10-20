"""
schema.py - single source of truth for field names, thresholds and display labels.
Everything here is available when a claim is FILED. Columns that only exist after the
claim is worked (fraud_score, claim_status, reserves, payments, LDFs) are NEVER used.
"""

TARGET = "siu_referral_flag"

# Injury severity (KABCO) -> ordinal
INJURY_ORD = {
    "O_no_injury": 0,
    "C_possible_injury": 1,
    "B_non_incapacitating": 2,
    "A_incapacitating": 3,
    "K_fatal": 4,
}
INJURY_LABEL = {v: k for k, v in INJURY_ORD.items()}

VEHICLE_CLASSES = [
    "suv_crossover", "sedan", "pickup_truck", "luxury_sedan",
    "economy_hatchback", "minivan", "sports_car", "electric_vehicle",
]

# ---- model inputs -----------------------------------------------------------
RAW_NUMERIC = [
    "age", "years_licensed", "mvr_points", "mvr_violations_3yr",
    "at_fault_accidents_3yr", "credit_score", "insurance_score", "annual_mileage",
    "vehicle_age_years", "actual_cash_value_usd", "gross_vehicle_damage_usd",
    "fnol_delay_days", "injuries_count_claimant", "vehicles_involved_count",
    "at_fault_pct_insured", "annual_premium_usd", "total_medical_cost_usd",
    "bi_gross_claim_usd", "ubi_score",
]
RAW_BOOL = [
    "attorney_represented", "police_report_filed", "hit_and_run_flag",
    "dui_flag", "telematics_enrolled",
]
DERIVED = [
    "injury_ord", "damage_ratio", "damage_vs_peer", "medical_vs_peer",
    "bi_vs_peer", "attorney_no_injury", "late_fnol", "no_police_serious",
]
FEATURE_COLS = RAW_NUMERIC + RAW_BOOL + DERIVED

# Must be present in an uploaded claim, otherwise no verdict is produced.
REQUIRED = [
    "injury_severity_code", "vehicle_class", "gross_vehicle_damage_usd",
    "actual_cash_value_usd", "attorney_represented", "police_report_filed",
    "fnol_delay_days",
]
# If missing and not required: bools default to False, numbers to the portfolio median,
# and a visible warning is always shown.
BOOL_DEFAULT_FALSE = ["hit_and_run_flag", "dui_flag", "telematics_enrolled"]

# ---- rule thresholds (set near the 95th percentile of the portfolio) --------------
LATE_FNOL_DAYS = 7
PEER_RATIO_FLAG = 3.0          # claimed amount vs peer median
REPEAT_AT_FAULT = 2
MIN_PEER_GROUP = 20            # smaller peer groups fall back to a broader group

# ---- decision tiers (percentile of historical out-of-fold risk scores) ----------
SIU_PERCENTILE = 97.0          # top 3% riskiest -> refer to SIU
REVIEW_PERCENTILE = 90.0       # top 10% -> request more documents

DISPLAY = {
    "claim_number": "Claim Number", "policy_id": "Policy ID",
    "vehicle_class": "Vehicle Class", "vehicle_make": "Vehicle Make",
    "vehicle_model": "Vehicle Model", "vehicle_age_years": "Vehicle Age (years)",
    "actual_cash_value_usd": "Actual Cash Value (USD)",
    "accident_type": "Accident Type", "vehicles_involved_count": "Vehicles Involved",
    "hit_and_run_flag": "Hit and Run", "police_report_filed": "Police Report Filed",
    "fnol_delay_days": "Reporting Delay (days)", "at_fault_pct_insured": "Insured At-Fault %",
    "injury_severity_code": "Injury Severity", "injuries_count_claimant": "Claimant Injuries",
    "total_medical_cost_usd": "Total Medical Cost (USD)",
    "bi_gross_claim_usd": "Bodily Injury Claim (USD)",
    "attorney_represented": "Attorney Represented",
    "gross_vehicle_damage_usd": "Gross Vehicle Damage (USD)",
    "age": "Driver Age", "years_licensed": "Years Licensed", "mvr_points": "MVR Points",
    "mvr_violations_3yr": "MVR Violations (3 yr)",
    "at_fault_accidents_3yr": "At-Fault Accidents (3 yr)",
    "credit_score": "Credit Score", "insurance_score": "Insurance Score",
    "annual_mileage": "Annual Mileage", "dui_flag": "DUI Flag",
    "annual_premium_usd": "Annual Premium (USD)",
    "telematics_enrolled": "Telematics Enrolled", "ubi_score": "UBI Score",
    # derived
    "injury_ord": "Injury severity level", "damage_ratio": "Damage vs vehicle value",
    "damage_vs_peer": "Repair cost vs peers", "medical_vs_peer": "Medical cost vs peers",
    "bi_vs_peer": "Injury claim vs peers", "attorney_no_injury": "Attorney on no-injury claim",
    "late_fnol": "Late reporting", "no_police_serious": "No police report on serious claim",
}


# ---- plain-English layer (what the user sees) -------------------------------------
APP_NAME = "ClaimGuard AI"
APP_TAGLINE = "Auto insurance claim risk checker"

ACT_APPROVE, ACT_REVIEW, ACT_INVESTIGATE = "Approve", "Review", "Investigate"

ACTION_UI = {
    ACT_APPROVE: {
        "icon": "✅", "title": "Approve",
        "tagline": "This claim looks normal. It can go ahead without extra checks.",
        "color": "#15803D", "bg": "#ECFDF3", "border": "#A7E3BE", "level": "Low risk", "pos": 16,
    },
    ACT_REVIEW: {
        "icon": "⚠️", "title": "Review before paying",
        "tagline": "A person should check a few things before this claim is paid.",
        "color": "#B45309", "bg": "#FFF7E6", "border": "#F6D08A", "level": "Medium risk", "pos": 50,
    },
    ACT_INVESTIGATE: {
        "icon": "🚨", "title": "Investigate for possible fraud",
        "tagline": "This claim has strong warning signs. Send it to the fraud investigation team.",
        "color": "#B91C1C", "bg": "#FEF0F0", "border": "#F4B4B4", "level": "High risk", "pos": 84,
    },
}

# How each model input is described to a non-expert
PLAIN_FEATURE = {
    "age": "the driver's age", "years_licensed": "how long the driver has been licensed",
    "mvr_points": "points on the driving record", "mvr_violations_3yr": "recent traffic violations",
    "at_fault_accidents_3yr": "recent at-fault accidents", "credit_score": "the driver's credit score",
    "insurance_score": "the driver's insurance score", "annual_mileage": "how many miles are driven each year",
    "vehicle_age_years": "the age of the vehicle", "actual_cash_value_usd": "the vehicle's market value",
    "gross_vehicle_damage_usd": "the vehicle repair cost", "fnol_delay_days": "how late the claim was reported",
    "injuries_count_claimant": "the number of injured people claimed",
    "vehicles_involved_count": "the number of vehicles involved",
    "at_fault_pct_insured": "how much of the blame is on the insured driver",
    "annual_premium_usd": "the policy's yearly premium", "total_medical_cost_usd": "the medical bills claimed",
    "bi_gross_claim_usd": "the injury payout requested", "ubi_score": "the driving-behaviour score",
    "attorney_represented": "a lawyer representing the claimant", "police_report_filed": "whether a police report exists",
    "hit_and_run_flag": "a hit-and-run", "dui_flag": "driving under the influence",
    "telematics_enrolled": "enrolment in the car-tracking programme", "injury_ord": "how serious the injuries are",
    "damage_ratio": "repair cost compared with the car's value",
    "damage_vs_peer": "repair cost compared with similar claims",
    "medical_vs_peer": "medical bills compared with similar claims",
    "bi_vs_peer": "the injury payout compared with similar claims",
    "attorney_no_injury": "a lawyer being involved when no injury was reported",
    "late_fnol": "late reporting", "no_police_serious": "no police report on a serious claim",
}

INJURY_PLAIN = {0: "No injury", 1: "Possible injury", 2: "Minor injury", 3: "Serious injury", 4: "Fatal"}
INJURY_PHRASE = {0: "no injury", 1: "a possible injury", 2: "a minor injury", 3: "a serious injury", 4: "a fatality"}
INJURY_CODE_PLAIN = {k: INJURY_PLAIN[v] for k, v in INJURY_ORD.items()}
