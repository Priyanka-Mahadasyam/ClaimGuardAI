import html
import io
import json
import os

import pandas as pd
import streamlit as st

from claim_parser import parse_claim_documents
from features import load_master
from pipeline import (
    MODEL_PATH, ask_ollama, build_prompt, fallback_explanation, load_bundle, score_claim,
)
from schema import (
    ACTION_UI, APP_NAME, APP_TAGLINE, INJURY_CODE_PLAIN, PLAIN_FEATURE,
)

st.set_page_config(page_title=f"{APP_NAME} | Claim risk checker", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="collapsed")

# ============================================================================ styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; height: 0; }
.block-container { max-width: 1440px; padding: 1.4rem clamp(1rem, 3vw, 3rem) 3.5rem; }
.stApp { background: radial-gradient(ellipse at 8% 0%, rgba(219,234,254,.55), transparent 32%), #F4F7FC; }
[data-testid="stAppViewContainer"] > .main { background: transparent; }
[data-testid="stVerticalBlock"] { gap: .85rem; }

.topbar { display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
  background:linear-gradient(115deg,#081A33 0%,#102B50 72%,#174477 100%); color:#fff; border:1px solid rgba(255,255,255,.08); box-shadow:0 12px 28px rgba(11,31,58,.14); border-radius:18px; padding:19px 24px; margin-bottom:18px; position:relative; overflow:hidden; }
.topbar:after { content:""; position:absolute; width:230px; height:230px; border-radius:50%; right:18%; top:-165px; background:rgba(56,189,248,.13); filter:blur(2px); pointer-events:none; }
.brand, .pill { position:relative; z-index:1; }
.brand { display:flex; align-items:center; gap:14px; }
.logo { font-size:26px; background:rgba(255,255,255,.12); border-radius:12px; width:46px; height:46px;
  display:flex; align-items:center; justify-content:center; }
.brand-name { font-size:20px; font-weight:800; letter-spacing:-.2px; }
.brand-tag { font-size:12.5px; color:#B6C4DA; margin-top:1px; }
.pill { font-size:12px; color:#CFE0FF; background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.18);
  border-radius:999px; padding:6px 12px; }

.hero { background:radial-gradient(circle at 88% 22%, rgba(255,255,255,.22), transparent 19%), linear-gradient(120deg,#153FC4 0%,#2563EB 53%,#22A8E8 100%); color:#fff; border:1px solid rgba(255,255,255,.35); box-shadow:0 16px 34px rgba(37,99,235,.18); border-radius:22px;
  padding:clamp(28px,4vw,48px); margin:20px 0 24px 0; position:relative; overflow:hidden; }
.hero:after { content:"✦"; position:absolute; right:5%; bottom:-68px; font-size:220px; line-height:1; color:rgba(255,255,255,.10); transform:rotate(-12deg); pointer-events:none; }
.hero h1, .hero p { position:relative; z-index:1; }
.hero h1 { font-size:34px; line-height:1.15; font-weight:800; margin:0 0 10px 0; letter-spacing:-.6px; color:#fff; }
.hero p { font-size:16px; line-height:1.55; margin:0; color:#E8F1FF; max-width:700px; }
.steps { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:0 0 22px 0; }
.step { background:linear-gradient(145deg,#fff,#F9FBFF); border:1px solid #E1E8F3; border-radius:17px; padding:20px 22px; box-shadow:0 5px 18px rgba(15,23,42,.035); transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease; }
.step:hover, .card:hover, .fact:hover, .stat:hover { transform:translateY(-2px); box-shadow:0 12px 26px rgba(15,23,42,.07); border-color:#C5D7F8; }
.step .num { width:30px; height:30px; border-radius:50%; background:#EAF1FF; color:#1D4ED8; font-weight:800;
  display:flex; align-items:center; justify-content:center; margin-bottom:10px; }
.step b { display:block; font-size:15.5px; color:#0F172A; margin-bottom:4px; }
.step span { font-size:13.5px; color:#64748B; line-height:1.5; }

.sec-title { font-size:18px; font-weight:700; color:#0F172A; margin:26px 0 10px 0; letter-spacing:-.2px; }
.sec-sub { font-size:13.5px; color:#64748B; margin:-6px 0 12px 0; }
.card { background:#fff; border:1px solid #E5E9F0; border-radius:16px; padding:20px 22px; }
.card h4 { margin:0 0 12px 0; font-size:15px; font-weight:700; color:#0F172A; }

.verdict { display:flex; justify-content:space-between; align-items:center; gap:28px; flex-wrap:wrap;
  border:1.5px solid; border-radius:20px; padding:26px 30px; margin:8px 0 14px 0; }
.v-left { display:flex; gap:18px; align-items:flex-start; flex:1 1 420px; }
.v-icon { font-size:40px; line-height:1; }
.v-kicker { font-size:11.5px; font-weight:700; letter-spacing:1.4px; margin-bottom:4px; }
.v-title { font-size:30px; font-weight:800; line-height:1.15; letter-spacing:-.5px; }
.v-tag { font-size:15.5px; color:#334155; margin-top:8px; line-height:1.5; }
.v-why { font-size:13.5px; color:#475569; margin-top:10px; line-height:1.5; }
.v-right { flex:0 0 260px; min-width:240px; }
.meter-label { font-size:13px; color:#475569; margin-bottom:8px; }
.meter { position:relative; display:flex; height:14px; border-radius:999px; overflow:hidden; }
.meter .g { flex:1; background:#34D399; } .meter .a { flex:1; background:#FBBF24; } .meter .r { flex:1; background:#F87171; }
.marker { position:absolute; top:-4px; width:22px; height:22px; margin-left:-11px; border-radius:50%;
  background:#fff; border:4px solid #0F172A; box-shadow:0 2px 6px rgba(15,23,42,.35); }
.meter-scale { display:flex; justify-content:space-between; font-size:11.5px; color:#64748B; margin-top:8px; }

.facts { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:6px; }
.fact { background:#fff; border:1px solid #E5E9F0; border-radius:14px; padding:14px 16px; }
.fact .k { font-size:12px; color:#64748B; margin-bottom:5px; }
.fact .v { font-size:19px; font-weight:700; color:#0F172A; letter-spacing:-.2px; }
.fact .s { font-size:12px; color:#64748B; margin-top:3px; }

.row { display:flex; gap:12px; align-items:flex-start; padding:11px 0; border-bottom:1px solid #EEF1F6; font-size:14.5px; color:#1E293B; line-height:1.45; }
.row:last-child { border-bottom:none; }
.dot { flex:0 0 10px; height:10px; border-radius:50%; margin-top:6px; }
.dot-bad { background:#DC2626; } .dot-warn { background:#F59E0B; } .dot-info { background:#3B82F6; } .dot-ok { background:#16A34A; }
.row .grow { flex:1; }
.step-item { display:flex; gap:12px; padding:10px 0; border-bottom:1px solid #EEF1F6; font-size:14.5px; color:#1E293B; line-height:1.45; }
.step-item:last-child { border-bottom:none; }
.step-n { flex:0 0 24px; height:24px; border-radius:50%; background:#0B1F3A; color:#fff; font-size:12px; font-weight:700;
  display:flex; align-items:center; justify-content:center; margin-top:1px; }
.chip { display:inline-block; font-size:12px; font-weight:600; padding:3px 10px; border-radius:999px; white-space:nowrap; }
.chip-ok { background:#DCFCE7; color:#166534; } .chip-warn { background:#FEF3C7; color:#92400E; }
.chip-bad { background:#FEE2E2; color:#991B1B; } .chip-info { background:#DBEAFE; color:#1E40AF; }

table.plain { width:100%; border-collapse:collapse; font-size:14.5px; }
table.plain th { text-align:left; font-size:12px; font-weight:600; color:#64748B; text-transform:uppercase; letter-spacing:.6px;
  padding:8px 10px; border-bottom:1px solid #E5E9F0; }
table.plain td { padding:13px 10px; border-bottom:1px solid #EEF1F6; color:#1E293B; }
table.plain tr:last-child td { border-bottom:none; }
.note { background:#EFF6FF; border:1px solid #BFDBFE; color:#1E3A8A; border-radius:12px; padding:11px 15px; font-size:13.5px; margin:10px 0; }
.muted { color:#64748B; font-size:13px; }
.stat-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:6px 0 16px 0; }
.stat { background:#fff; border:1px solid #E5E9F0; border-radius:16px; padding:20px 22px; }
.stat .big { font-size:34px; font-weight:800; color:#1D4ED8; letter-spacing:-.8px; }
.stat .lbl { font-size:13.5px; color:#475569; margin-top:4px; line-height:1.45; }
.footer { text-align:center; color:#94A3B8; font-size:12.5px; margin-top:38px; padding-top:18px; border-top:1px solid #E5E9F0; line-height:1.6; }

[data-testid="stFileUploaderDropzone"] { background:#fff; border:2px dashed #93B4F8; border-radius:16px; padding:26px; }
[data-testid="stFileUploaderDropzone"]:hover { border-color:#1D4ED8; background:#F5F9FF; }
.stButton > button { border-radius:11px; border:1px solid #CBD5E1; font-weight:600; min-height:42px; transition:all .18s ease; box-shadow:0 1px 2px rgba(15,23,42,.03); }
.stButton > button[kind="primary"] { background:linear-gradient(110deg,#1D4ED8,#2687E8); border-color:#2563EB; color:white; }
[data-testid="stSegmentedControl"] { background:#E9EFF8; border:1px solid #D8E2F0; padding:4px; border-radius:13px; width:fit-content; }
[data-testid="stSegmentedControl"] button { border-radius:9px !important; font-weight:600 !important; }
[data-testid="stDataFrame"] { border:1px solid #DCE4F0; border-radius:13px; overflow:hidden; box-shadow:0 5px 16px rgba(15,23,42,.035); }
[data-testid="stSelectbox"] [data-baseweb="select"] > div { border-radius:11px; border-color:#CBD5E1; }
[data-testid="stFileUploaderDropzone"] { box-shadow:inset 0 0 0 1px rgba(255,255,255,.75), 0 8px 24px rgba(37,99,235,.05); }
.stButton > button:hover { border-color:#1D4ED8; color:#1D4ED8; }
@media (max-width: 820px) { .block-container { padding-left:1rem; padding-right:1rem; } .facts { grid-template-columns:repeat(2,1fr); } .steps, .stat-grid { grid-template-columns:1fr; }
  .hero h1 { font-size:26px; } .v-title { font-size:24px; } }
h1 a, h2 a, h3 a, h4 a, [data-testid="stHeaderActionElements"] { display:none !important; }
table.plain, table.plain th, table.plain td { border-left:none !important; border-right:none !important; border-top:none !important; background:transparent !important; }
table.plain th, table.plain td { border-bottom:1px solid #EEF1F6 !important; }
table.plain tr:last-child td { border-bottom:none !important; }
.card, .fact, .stat, .step { box-shadow:0 1px 2px rgba(15,23,42,.04); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ============================================================================ helpers
def esc(t) -> str:
    """HTML-escape, and neutralise $ so Streamlit does not turn it into math."""
    return html.escape(str(t)).replace("$", "&#36;")


def md(text: str) -> str:
    return text.replace("$", "\\$")


def show(markup: str):
    st.markdown(" ".join(line.strip() for line in markup.strip().splitlines()), unsafe_allow_html=True)


def chip(text: str, level: str) -> str:
    return f'<span class="chip chip-{level}">{esc(text)}</span>'


def money(v) -> str:
    return f"${float(v):,.0f}"


def section(title: str, sub: str = ""):
    show(f'<div class="sec-title">{esc(title)}</div>' + (f'<div class="sec-sub">{esc(sub)}</div>' if sub else ""))


# ============================================================================ resources
if not os.path.exists(MODEL_PATH):
    # first start (for example on a fresh server): build the risk model automatically
    with st.spinner("First-time setup: building the risk model from the claims data (about a minute)..."):
        try:
            import train_model
            train_model.main()
        except Exception as e:
            st.error(f"The risk model could not be built: {e}. Check that the 3 data files are in the data/ folder.")
            st.stop()
    st.rerun()


@st.cache_resource
def get_bundle():
    return load_bundle()


@st.cache_data
def get_master():
    return load_master("data")


bundle = get_bundle()

SAMPLES = [
    ("Typical fender-bender", "claim_1_clean_typical.pdf"),
    ("Lawyer, but no injury", "claim_2_attorney_no_injury.pdf"),
    ("Missing police report", "claim_3_missing_police_report.pdf"),
    ("Very large injury claim", "claim_4_large_legit_fatal.pdf"),
    ("Inflated repair bill", "claim_5_repair_cost_outlier.pdf"),
    ("Incomplete form", "claim_6_incomplete_form.pdf"),
]


class _Doc(io.BytesIO):
    """Stored document bytes that behave like an uploaded file."""
    def __init__(self, name, data):
        super().__init__(data)
        self.name = name


# ============================================================================ result view
def render_result(result: dict, key: str, note: str = ""):
    claim = result["claim"]
    ui = ACTION_UI[result["action"]]

    show(f"""
    <div class="verdict" style="background:{ui['bg']};border-color:{ui['border']}">
      <div class="v-left">
        <div class="v-icon">{ui['icon']}</div>
        <div>
          <div class="v-kicker" style="color:{ui['color']}">RECOMMENDED DECISION</div>
          <div class="v-title" style="color:{ui['color']}">{esc(ui['title'])}</div>
          <div class="v-tag">{esc(ui['tagline'])}</div>
          <div class="v-why"><b>Why:</b> {esc(result['action_reason'])}.</div>
        </div>
      </div>
      <div class="v-right">
        <div class="meter-label">Overall risk: <b style="color:{ui['color']}">{esc(ui['level'])}</b></div>
        <div class="meter"><div class="g"></div><div class="a"></div><div class="r"></div>
          <div class="marker" style="left:{ui['pos']}%"></div></div>
        <div class="meter-scale"><span>Low</span><span>Medium</span><span>High</span></div>
      </div>
    </div>""")
    if note:
        st.caption(note)

    bi = float(claim.get("bi_gross_claim_usd") or 0)
    injury = INJURY_CODE_PLAIN.get(claim.get("injury_severity_code"), "-")
    show(f"""
    <div class="facts">
      <div class="fact"><div class="k">Claim number</div><div class="v">{esc(claim.get('claim_number') or '-')}</div>
        <div class="s">{esc(str(claim.get('vehicle_class', '')).replace('_', ' ').title())}</div></div>
      <div class="fact"><div class="k">Repair cost claimed</div><div class="v">{money(claim['gross_vehicle_damage_usd'])}</div>
        <div class="s">Injury level: {esc(injury)}</div></div>
      <div class="fact"><div class="k">Injury payout claimed</div><div class="v">{money(bi) if bi > 0 else 'None'}</div>
        <div class="s">{'Lawyer involved' if claim.get('attorney_represented') else 'No lawyer involved'}</div></div>
    </div>""")

    if result["filled_fields"]:
        show('<div class="note">ℹ️ Some details were missing from the document, so typical values were used for: '
             + esc(", ".join(result["filled_fields"])) + ".</div>")

    # ---- why + next steps
    section("Why this result")
    left, right = st.columns([3, 2], gap="medium")
    with left:
        rows = ""
        for f in result["warning_signs"]:
            lvl = {"high": "bad", "review": "warn", "docs": "warn"}.get(f["severity"], "info")
            rows += f'<div class="row"><span class="dot dot-{lvl}"></span><div class="grow">{esc(f["text"])}</div></div>'
        if not result["warning_signs"]:
            rows = '<div class="row"><span class="dot dot-ok"></span><div class="grow">No warning signs were found on this claim.</div></div>'
        extra = ""
        if result["reasons"]:
            lv = {"Strong": "bad", "Moderate": "warn", "Slight": "info"}
            extra = '<h4 style="margin-top:18px">Also similar to past investigated claims</h4>' + "".join(
                f'<div class="row"><div class="grow">{esc(r["text"])}</div>{chip(r["strength"], lv[r["strength"]])}</div>'
                for r in result["reasons"])
        show(f'<div class="card"><h4>Warning signs</h4>{rows}{extra}</div>')
    with right:
        items = "".join(f'<div class="step-item"><span class="step-n">{i}</span><div>{esc(s)}</div></div>'
                        for i, s in enumerate(result["next_steps"], 1))
        show(f'<div class="card"><h4>What to do next</h4>{items}</div>')

    # ---- comparison
    section("How this claim compares with similar claims",
            f"Similar claims = {result['peer_group']}.")
    body = "".join(
        f'<tr><td><b>{esc(r["check"])}</b></td><td>{esc(r["claim"])}</td><td>{esc(r["typical"])}</td>'
        f'<td>{chip(r["status"], r["level"])} <span class="muted">&nbsp;{r["ratio"]:.1f}× typical</span></td></tr>'
        for r in result["peer_rows"])
    show(f'<div class="card"><table class="plain"><tr><th>What we checked</th><th>This claim</th>'
         f'<th>Typical for similar claims</th><th>Result</th></tr>{body}</table></div>')

    # ---- AI summary
    section("Plain-English summary")
    if st.button("✨ Explain this claim in simple words", key=f"llm_{key}"):
        with st.spinner("Writing the summary..."):
            try:
                text = ask_ollama(build_prompt(result))
            except Exception:
                text = fallback_explanation(result)
                st.caption("The AI assistant is not running, so a built-in summary is shown instead.")
        with st.container(border=True):
            st.markdown(md(text))

    # ---- technical details: avoid presenting probability estimates or local driver
    # values as percentage-point contributions unless they are properly calibrated.
    with st.expander("Technical details (for analysts)"):
        t = result["technical"]
        st.write(f"**Model used:** {t['model_name']}")
        st.caption(
            "The model score helps prioritise claims for review. It is not a probability "
            "that fraud occurred, and it should not be treated as proof."
        )
        if t["drivers"]:
            st.write("Factors that influenced this claim's model score:")
            driver_names = []
            for d in t["drivers"]:
                name = str(d.get("feature", "")).strip()
                if name and name not in driver_names:
                    driver_names.append(name)
            if driver_names:
                st.dataframe(pd.DataFrame({"Factor": driver_names}), hide_index=True, use_container_width=True)
            else:
                st.write("No individual factors are available to display.")
        else:
            st.write("No individual factors are available to display for this claim.")


# ============================================================================ header + navigation
show(f"""
<div class="topbar">
  <div class="brand"><div class="logo">🛡️</div>
    <div><div class="brand-name">{esc(APP_NAME)}</div><div class="brand-tag">{esc(APP_TAGLINE)}</div></div></div>
  <div class="pill">Decision support · Demo data</div>
</div>""")

NAV = ["Check a claim", "Past claims", "How accurate is it?"]
page = st.segmented_control("Navigation", NAV, default=NAV[0], label_visibility="collapsed") or NAV[0]

# ============================================================================ page 1
if page == NAV[0]:
    ss = st.session_state
    ss.setdefault("doc", None)       # (file name, bytes) of the claim being checked
    ss.setdefault("up_ver", 0)

    if ss["doc"] is None:
        show("""
        <div class="hero"><h1>Check an insurance claim in seconds</h1>
        <p>Upload a claim form and get a clear recommendation: approve it, review it, or send it for investigation.
        You also get the reasons in plain English and what to check next.</p></div>""")
        up = st.file_uploader("Upload one or more claims (PDF, Excel, CSV or JSON)", type=["pdf", "xlsx", "xls", "json", "csv"],
                              key=f"uploader_{ss['up_ver']}",
                              help="PDFs can contain one claim per page. Excel/CSV/JSON can contain one claim per row/object. PDF forms use labelled lines; tabular files use the dataset column names.")
        if up is not None:
            ss["doc"] = (up.name, up.getvalue())
            st.rerun()

        if os.path.isdir("sample_claims"):
            section("Or try an example", "Click any example to see how the system reacts.")
            cols = st.columns(3)
            for i, (label, fname) in enumerate(SAMPLES):
                path = os.path.join("sample_claims", fname)
                if os.path.exists(path) and cols[i % 3].button(label, key=f"sample_{i}", use_container_width=True):
                    ss["doc"] = (fname, open(path, "rb").read())
                    st.rerun()

        show("""
        <div class="sec-title">How it works</div>
        <div class="steps">
          <div class="step"><div class="num">1</div><b>Upload the claim</b><span>Drop in the claim form. We read the details for you.</span></div>
          <div class="step"><div class="num">2</div><b>We compare it</b><span>The claim is checked against thousands of similar past claims.</span></div>
          <div class="step"><div class="num">3</div><b>You decide</b><span>You see a recommendation, the warning signs and the next steps.</span></div>
        </div>""")
    else:
        name, data = ss["doc"]
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        c1.markdown(f"**Checking:** {md(name)}")
        if c2.button("↺ New claim", use_container_width=True):
            ss["doc"] = None
            ss["up_ver"] += 1
            st.rerun()

        try:
            parsed_claims = parse_claim_documents(_Doc(name, data))
        except Exception as e:
            st.error(f"We could not open this file: {e}")
            st.stop()

        if len(parsed_claims) > 1:
            st.success(f"Detected {len(parsed_claims)} claims. Each claim will be assessed separately.")
            summary_rows = []
            for i, parsed in enumerate(parsed_claims, start=1):
                claim_label = parsed.get("claim_number") or f"Claim {i}"
                if parsed.get("missing_required"):
                    summary_rows.append({"Claim": claim_label, "Decision": "Could not assess",
                                         "Risk": "Missing required fields",
                                         "Details missing": ", ".join(parsed["missing_required"])})
                else:
                    result = score_claim(parsed, bundle)
                    summary_rows.append({"Claim": claim_label, "Decision": result["action"],
                                         "Risk": ACTION_UI[result["action"]]["level"],
                                         "Details missing": "None"})
            st.subheader("Batch summary")
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
            st.divider()

        for i, parsed in enumerate(parsed_claims, start=1):
            claim_label = parsed.get("claim_number") or f"Claim {i}"
            if len(parsed_claims) > 1:
                st.subheader(f"Claim {i}: {claim_label}")
            for n in parsed.get("notes", []):
                show(f'<div class="note">ℹ️ {esc(n)}</div>')
            if parsed["missing_required"]:
                st.error(f"{claim_label}: We could not find these required details: "
                         + ", ".join(parsed["missing_required"])
                         + ". This claim was not scored, but the other claims can still be assessed.")
                if parsed.get("raw_text"):
                    with st.expander(f"See readable text for {claim_label}"):
                        st.text(parsed["raw_text"])
                continue
            render_result(score_claim(parsed, bundle), key=f"upload_{i}")
            if i < len(parsed_claims):
                st.divider()

# ============================================================================ page 2
elif page == NAV[1]:
    master = get_master()
    oof = bundle["oof"].set_index("claim_number")["oof_score"]
    m = master.assign(oof_score=master["claim_number"].map(oof))

    section("Past claims", "Browse the 5,000 claims the system learned from. Each one is scored by a version "
            "of the system that never saw that claim, so the results are fair.")
    investigated_count = int(m["siu_referral_flag"].fillna(False).astype(bool).sum())
    only = st.checkbox(
        f"Only show claims that investigators actually looked into ({investigated_count:,})",
        key="past_claims_investigated_only",
    )
    view = m[m["siu_referral_flag"].fillna(False).astype(bool)] if only else m
    ranked = view.sort_values("oof_score", ascending=False).reset_index(drop=True)

    # Display every matching row. The fixed-height table scrolls internally,
    # rather than silently truncating the results to the first 12 claims.
    table = pd.DataFrame({
        "Claim": ranked["claim_number"],
        "Vehicle": ranked["vehicle_class"].str.replace("_", " ").str.title(),
        "Injury level": ranked["injury_severity_code"].map(INJURY_CODE_PLAIN),
        "Lawyer involved": ranked["attorney_represented"].map({True: "Yes", False: "No"}),
        "Repair cost": ranked["gross_vehicle_damage_usd"].map(money),
        "Investigated in the past": ranked["siu_referral_flag"].map({True: "Yes", False: "No"}),
    })
    st.caption(f"Showing {len(table):,} of {len(m):,} claims. Scroll inside the table to browse all results.")
    st.dataframe(table, hide_index=True, use_container_width=True, height=480)

    section("Open a claim")
    # Keep every available historical claim selectable, not just the first 300.
    # Streamlit's selectbox supports typing to search through the options.
    claim_options = ranked["claim_number"].dropna().astype(str).tolist()
    st.caption(f"{len(claim_options):,} claims available. Click the dropdown and type a claim number to search.")
    pick = st.selectbox("Choose a claim", claim_options, label_visibility="collapsed")
    row = m[m["claim_number"] == pick].iloc[0]
    st.markdown(f"**What actually happened:** {'investigators looked into this claim' if row['siu_referral_flag'] else 'no investigation'}")
    render_result(score_claim(row.to_dict(), bundle, raw_override=float(row["oof_score"])), key="browse",
                  note="This is how the system would have judged this claim without having seen it before.")

# ============================================================================ page 3
else:
    path = os.path.join("models", "metrics.json")
    if not os.path.exists(path):
        st.error("Accuracy report not found. Run `python train_model.py`.")
        st.stop()
    r = json.load(open(path))
    sel = r["candidates"][r["selected_model"]]["mean"]

    section("How well does it work?", "A quick overview of the demo's claim-risk model.")
    show(f"""
    <div class="stat-grid">
      <div class="stat"><div class="big">{r['n_claims']:,}</div><div class="lbl">claims included in the dataset</div></div>
      <div class="stat"><div class="big">{r['n_positive']}</div><div class="lbl">past claims sent for investigation ({r['base_rate_pct']}%)</div></div>
      <div class="stat"><div class="big">Human review</div><div class="lbl">the system flags risk; it does not decide fraud</div></div>
    </div>
    <div class="card">
      <h4>How to understand the results</h4>
      <div class="row"><span class="dot dot-info"></span><div class="grow"><b>Higher risk means review first.</b> The score helps adjusters prioritise claims that may need a closer look.</div></div>
      <div class="row"><span class="dot dot-info"></span><div class="grow"><b>A risk flag is not proof of fraud.</b> A person must check the claim and make the final decision.</div></div>
      <div class="row"><span class="dot dot-warn"></span><div class="grow"><b>This is a demo, not a real-world accuracy guarantee.</b> The dataset is simulated and contains only {r['n_positive']} past investigation cases.</div></div>
    </div>""")

    with st.expander("Technical results (optional)"):
        base = r["base_rate_pct"] / 100
        recall = sel["recall_at_budget"]
        prec = sel["precision_at_budget"]
        a, b, c, d = st.columns(4)
        a.metric("ROC-AUC", f"{sel['roc_auc']:.2f}")
        b.metric("PR-AUC", f"{sel['pr_auc']:.2f}")
        c.metric("Recall at review limit", f"{recall:.0%}")
        d.metric("Precision at review limit", f"{prec:.0%}")
        st.caption("These are model evaluation metrics on held-out claims. They are useful for technical review, but they do not prove real-world fraud accuracy.")

# ============================================================================ footer
show(f"""<div class="footer">{esc(APP_NAME)} is a decision-support tool. A human adjuster always makes the final decision.<br>
Demo built on synthetic data (XpertSystems INS-006 sample, CC-BY-NC-4.0).</div>""")
