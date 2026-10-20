"""
claim_parser.py - reads an uploaded claim (PDF / Excel / CSV / JSON) into one dict.

Rules:
  * No silent guessing. Required fields that cannot be read are reported in
    `missing_required` and the app stops instead of producing a verdict.
  * CSV / Excel / JSON use the dataset's own column names (e.g. gross_vehicle_damage_usd).
  * PDF uses "Label: value" lines (aliases below), and also handles a label on one line
    with its value on the next line.
"""

import json
import re

import pandas as pd

from schema import DISPLAY, INJURY_ORD, REQUIRED, VEHICLE_CLASSES

# key -> (type, [label aliases])
FIELD_SPECS = {
    "claim_number": ("str", ["Claim Number", "Claim ID", "Claim No"]),
    "policy_id": ("str", ["Policy ID", "Policy Number"]),
    "vehicle_class": ("vclass", ["Vehicle Class", "Vehicle Type", "Body Style"]),
    "vehicle_make": ("str", ["Vehicle Make", "Make"]),
    "vehicle_model": ("str", ["Vehicle Model", "Model"]),
    "vehicle_age_years": ("num", ["Vehicle Age", "Age of Vehicle"]),
    "actual_cash_value_usd": ("num", ["Actual Cash Value", "Vehicle ACV", "ACV"]),
    "accident_type": ("str", ["Accident Type", "Collision Type"]),
    "vehicles_involved_count": ("num", ["Vehicles Involved", "Number of Vehicles"]),
    "hit_and_run_flag": ("bool", ["Hit and Run", "Hit-and-Run"]),
    "police_report_filed": ("bool", ["Police Report Filed", "Police Report"]),
    "fnol_delay_days": ("num", ["Reporting Delay", "Days to Report", "FNOL Delay"]),
    "at_fault_pct_insured": ("num", ["Insured At-Fault %", "At-Fault %", "Insured At Fault Percent"]),
    "injury_severity_code": ("injury", ["Injury Severity", "Claimant Injury Severity"]),
    "injuries_count_claimant": ("num", ["Claimant Injuries", "Injured Claimants"]),
    "total_medical_cost_usd": ("num", ["Total Medical Cost", "Medical Costs"]),
    "bi_gross_claim_usd": ("num", ["Bodily Injury Claim", "BI Claim Amount"]),
    "attorney_represented": ("bool", ["Attorney Represented", "Attorney Involved", "Legal Representation"]),
    "gross_vehicle_damage_usd": ("num", ["Gross Vehicle Damage", "Vehicle Damage", "Repair Estimate"]),
    "age": ("num", ["Driver Age", "Insured Age"]),
    "years_licensed": ("num", ["Years Licensed"]),
    "mvr_points": ("num", ["MVR Points"]),
    "mvr_violations_3yr": ("num", ["MVR Violations"]),
    "at_fault_accidents_3yr": ("num", ["At-Fault Accidents", "At Fault Accidents"]),
    "credit_score": ("num", ["Credit Score"]),
    "insurance_score": ("num", ["Insurance Score"]),
    "annual_mileage": ("num", ["Annual Mileage"]),
    "dui_flag": ("bool", ["DUI Flag", "DUI"]),
    "annual_premium_usd": ("num", ["Annual Premium"]),
    "telematics_enrolled": ("bool", ["Telematics Enrolled", "Telematics"]),
    "ubi_score": ("num", ["UBI Score"]),
}

_YES = {"yes", "y", "true", "1", "present", "filed", "attached"}
_NO = {"no", "n", "false", "0", "absent", "none", "not filed"}
_NULLS = {"", "n/a", "na", "none", "null", "-", "nan", "not provided", "unknown"}


# ---------------------------------------------------------------- value parsers
def _num(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return None if pd.isna(v) else float(v)
    s = str(v).strip()
    if s.lower() in _NULLS:
        return None
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", s)
    return float(m.group(0).replace(",", "")) if m else None


def _bool(v):
    if isinstance(v, bool):
        return v
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    words = re.findall(r"[a-z0-9]+", str(v).lower())
    if not words:
        return None
    if words[0] in _YES:
        return True
    if words[0] in _NO:
        return False
    if " ".join(words[:2]) in _NO:
        return False
    return None


def _injury(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if s in INJURY_ORD:
        return s
    low = s.lower()
    if low in _NULLS:
        return None
    m = re.match(r"^([OCBAK])(?:[_\s\-]|$)", s, re.I)
    if m:
        letter = m.group(1).upper()
        return next(k for k in INJURY_ORD if k.startswith(letter + "_"))
    if "non" in low and "incap" in low:
        return "B_non_incapacitating"
    if "incap" in low:
        return "A_incapacitating"
    if "fatal" in low or "death" in low:
        return "K_fatal"
    if "possible" in low:
        return "C_possible_injury"
    if "no injury" in low or "none" in low or "uninjured" in low:
        return "O_no_injury"
    return None


def _vclass(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = re.sub(r"[^a-z0-9]+", "_", str(v).strip().lower()).strip("_")
    if s in _NULLS:
        return None
    if s in VEHICLE_CLASSES:
        return s
    for key, target in [("suv", "suv_crossover"), ("crossover", "suv_crossover"), ("pickup", "pickup_truck"),
                        ("truck", "pickup_truck"), ("minivan", "minivan"), ("van", "minivan"),
                        ("sport", "sports_car"), ("luxury", "luxury_sedan"), ("electric", "electric_vehicle"),
                        ("ev", "electric_vehicle"), ("hatch", "economy_hatchback"), ("economy", "economy_hatchback"),
                        ("sedan", "sedan")]:
        if key in s.split("_") or (len(key) > 3 and key in s):
            return target
    return s  # unknown class: kept as-is, pipeline falls back to broader peers


def _str(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return None if s.lower() in _NULLS else s


_CAST = {"num": _num, "bool": _bool, "injury": _injury, "vclass": _vclass, "str": _str}


# ---------------------------------------------------------------- PDF text
def _find_in_lines(lines, aliases):
    for alias in sorted(aliases, key=len, reverse=True):
        pat = re.compile(rf"^\s*{re.escape(alias)}(?:\s*\([^)]*\))?\s*:\s*(.+?)\s*$", re.I)
        for ln in lines:
            m = pat.match(ln)
            if m:
                return m.group(1)
    for alias in sorted(aliases, key=len, reverse=True):  # table layout: label, then value on next line
        pat = re.compile(rf"^\s*{re.escape(alias)}(?:\s*\([^)]*\))?\s*$", re.I)
        for i, ln in enumerate(lines[:-1]):
            if pat.match(ln):
                return lines[i + 1]
    return None


def parse_pdf_text(raw: str) -> dict:
    text = raw.replace("\r", "").replace("\u00a0", " ")
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    claim = {}
    for key, (kind, aliases) in FIELD_SPECS.items():
        claim[key] = _CAST[kind](_find_in_lines(lines, aliases))
    return _finish(claim, raw_text=text)


# ---------------------------------------------------------------- tabular
def _norm_key(k):
    return re.sub(r"[^a-z0-9]+", "_", str(k).strip().lower()).strip("_")


def parse_record(rec: dict) -> dict:
    rec = {_norm_key(k): v for k, v in rec.items()}
    claim = {}
    for key, (kind, _) in FIELD_SPECS.items():
        claim[key] = _CAST[kind](rec.get(key))
    return _finish(claim)


def _finish(claim: dict, raw_text: str = "") -> dict:
    missing = [DISPLAY[k] for k in REQUIRED if claim.get(k) is None]
    claim["missing_required"] = missing
    claim["raw_text"] = raw_text
    return claim


# ---------------------------------------------------------------- entry points

def parse_claim_documents(file) -> list[dict]:
    """Parse every claim in an uploaded document.

    PDF: one claim per page, and also supports repeated Claim Number sections on a page.
    CSV / Excel / JSON: one claim per non-empty row/object.
    """
    name = file.name.lower()
    notes = []
    results = []

    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(file)
        page_texts = [(page.extract_text() or "").replace("\r", "") for page in reader.pages]
        if not any(text.strip() for text in page_texts):
            res = _finish({k: None for k in FIELD_SPECS})
            res["notes"] = ["No readable text found - this looks like a scanned PDF (OCR is not supported)."]
            return [res]

        # Usually each claim occupies one page. If multiple claim-number labels appear
        # on a page, split at each label so fields from separate claims aren't mixed.
        claim_number_re = re.compile(r"(?im)^\s*Claim(?:\s+Number|\s+ID|\s+No)(?:\s*:\s*.*)?$")
        chunks = []
        for page_no, text in enumerate(page_texts, start=1):
            if not text.strip():
                continue
            matches = list(claim_number_re.finditer(text))
            if len(matches) > 1:
                starts = [m.start() for m in matches] + [len(text)]
                for i in range(len(starts)-1):
                    chunk = text[starts[i]:starts[i+1]].strip()
                    if chunk:
                        chunks.append((page_no, chunk))
            else:
                chunks.append((page_no, text.strip()))

        for page_no, chunk in chunks:
            # Ignore pages that clearly contain no claim fields.
            if not re.search(r"(?im)^\s*(Claim Number|Claim ID|Claim No|Vehicle Class|Vehicle Type)\s*(?::|$)", chunk):
                continue
            res = parse_pdf_text(chunk)
            res["notes"] = []
            res["source_page"] = page_no
            results.append(res)

        if not results:
            res = _finish({k: None for k in FIELD_SPECS}, raw_text="\n".join(page_texts))
            res["notes"] = ["Could not identify a claim form in the PDF. Ensure claim fields use labels such as 'Claim Number:' and 'Vehicle Class:'."]
            return [res]
        if len(results) > 1:
            for i, res in enumerate(results, start=1):
                res["notes"].append(f"Claim {i} extracted from PDF page {res.get('source_page')}.")
        return results

    if name.endswith(".csv"):
        df = pd.read_csv(file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    elif name.endswith(".json"):
        data = json.load(file)
        df = pd.DataFrame(data if isinstance(data, list) else [data])
    else:
        raise ValueError("Unsupported file format. Use PDF, Excel, CSV or JSON.")

    if df.empty:
        raise ValueError("The uploaded file has no rows.")

    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("The uploaded file has no non-empty claim rows.")
    for row_index, (_, row) in enumerate(df.iterrows(), start=1):
        res = parse_record(row.to_dict())
        res["notes"] = []
        res["source_row"] = row_index
        if len(df) > 1:
            res["notes"].append(f"Claim {row_index} extracted from data row {row_index}.")
        results.append(res)
    return results


def parse_claim_document(file) -> dict:
    """Backward-compatible single-claim parser; returns the first parsed claim."""
    results = parse_claim_documents(file)
    if not results:
        raise ValueError("No claims were found in the uploaded file.")
    if len(results) > 1:
        results[0].setdefault("notes", []).append(
            f"This file contains {len(results)} claims; this legacy function returns only the first. "
            "Use parse_claim_documents() to process every claim."
        )
    return results[0]
