"""
app.py
------
Placement Prediction — Streamlit app built around the uploaded
LogisticRegression model (placement_model.pkl / pipeline_bundle.pkl).

Run:
    streamlit run app.py

Requires pipeline_bundle.pkl in the same folder (built by build_pipeline.py),
and the .streamlit/config.toml folder sitting alongside this file.
"""

import joblib
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Placement Readiness",
    page_icon="\U0001F393",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Design system: "Transcript & Seal" — a cream ledger page with a brass
# wax-seal medallion that renders the model's probability as the one bold
# signature moment, held in a real st.container so it always nests correctly.
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --ink: #0F1B3C;
    --ink-2: #1B2A4A;
    --parchment: #F7F3E8;
    --parchment-2: #EFE8D8;
    --brass: #C9A227;
    --brass-light: #9A7B1A;
    --placed: #2E7D63;
    --not-placed: #B5533C;
    --rule: rgba(15, 27, 60, 0.16);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: var(--parchment);
}

.block-container {
    max-width: 760px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

/* Header */
.masthead { text-align: center; margin-bottom: 1.6rem; }
.eyebrow {
    font-size: 0.72rem;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: var(--brass-light);
    margin-bottom: 0.6rem;
    font-weight: 600;
}
.masthead h1 {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.6rem;
    color: var(--ink);
    margin: 0;
    line-height: 1.1;
}
.masthead .sub {
    font-family: 'Fraunces', serif;
    font-style: italic;
    font-weight: 400;
    color: rgba(15,27,60,0.55);
    font-size: 1.02rem;
    margin-top: 0.5rem;
}
.rule {
    width: 72px;
    height: 2px;
    background: var(--brass);
    margin: 1.1rem auto 0 auto;
    opacity: 0.9;
}

.section-label {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--ink);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.6rem 0 0.9rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--rule);
}

/* Widget labels */
[data-testid="stWidgetLabel"] p {
    font-weight: 600 !important;
    color: var(--ink) !important;
    font-size: 0.85rem !important;
}

/* Selectbox */
div[data-baseweb="select"] > div {
    background-color: white !important;
    border-radius: 4px !important;
    border: 1px solid rgba(15,27,60,0.25) !important;
}
div[data-baseweb="select"] * { color: var(--ink) !important; }
ul[data-baseweb="menu"] { background-color: white !important; }
ul[data-baseweb="menu"] li, ul[data-baseweb="menu"] li * { color: var(--ink) !important; }

/* Slider thumb + value readout, with a brass fallback in case the
   config.toml primaryColor isn't picked up by the runtime */
[data-testid="stSlider"] [role="slider"] {
    background-color: var(--brass) !important;
    border-color: var(--brass) !important;
}
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: rgba(15,27,60,0.55) !important;
}
[data-testid="stThumbValue"] { color: var(--ink) !important; font-weight: 700 !important; }

/* Radio buttons */
[data-testid="stRadio"] label p { color: var(--ink) !important; }

.stButton > button {
    width: 100%;
    background: var(--ink);
    color: var(--parchment);
    border: 1px solid var(--brass);
    border-radius: 4px;
    padding: 0.75rem 1rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-size: 0.82rem;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: var(--brass);
    color: var(--ink);
    border-color: var(--brass);
}

/* Result panel — a real st.container(key="result_panel"), so this class
   reliably wraps the seal + verdict + factor bars in one dark card. */
.st-key-result_panel {
    background: var(--ink);
    border-radius: 6px;
    padding: 2rem 1.6rem 1.6rem 1.6rem;
    margin-top: 1.8rem;
    box-shadow: 0 20px 45px rgba(15,27,60,0.25);
}
.seal { width: 168px; height: 168px; margin: 0 auto 1.1rem auto; }
.verdict {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 1.7rem;
    text-align: center;
    margin: 0.2rem 0 0.3rem 0;
}
.verdict.placed { color: var(--placed); }
.verdict.notplaced { color: var(--not-placed); }
.verdict-sub {
    text-align: center;
    color: rgba(247,243,232,0.65);
    font-size: 0.88rem;
    margin-bottom: 0.6rem;
}
.factor-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: rgba(247,243,232,0.9);
    padding: 0.45rem 0;
    border-bottom: 1px solid rgba(247,243,232,0.14);
}
.factor-row .fname { color: rgba(247,243,232,0.6); flex: 0 0 130px; }
.factor-row .fbar-wrap {
    flex: 1;
    margin: 0 0.9rem;
    background: rgba(247,243,232,0.12);
    border-radius: 3px;
    height: 6px;
    align-self: center;
    position: relative;
    overflow: hidden;
}
.factor-row .fbar {
    position: absolute;
    top: 0; left: 0; bottom: 0;
    border-radius: 3px;
}
.footnote {
    text-align: center;
    color: rgba(15,27,60,0.4);
    font-size: 0.72rem;
    margin-top: 2rem;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_bundle():
    return joblib.load("pipeline_bundle.pkl")


bundle = load_bundle()
model = bundle["model"]
scaler = bundle["scaler"]
feature_cols = bundle["feature_cols"]
student_id_fill = bundle["student_id_fill"]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="masthead">
        <div class="eyebrow">Campus Placement Office</div>
        <h1>Placement Readiness</h1>
        <div class="sub">an estimate, not a verdict</div>
        <div class="rule"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Inputs — plain sections directly on the parchment page (no fake card div,
# so nothing can silently detach from its intended styling).
# ---------------------------------------------------------------------------
st.markdown('<div class="section-label">Academic Record</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    cgpa = st.slider("CGPA", 6.0, 9.5, 7.7, 0.1)
    ssc_marks = st.slider("SSC Marks (%)", 50, 95, 70)
with c2:
    hsc_marks = st.slider("HSC Marks (%)", 50, 95, 75)
    aptitude = st.slider("Aptitude Test Score", 40, 100, 70)

st.markdown('<div class="section-label">Experience & Skills</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    internships = st.selectbox("Internships completed", [0, 1, 2, 3], index=1)
    projects = st.selectbox("Projects completed", [0, 1, 2, 3, 4, 5], index=2)
with c4:
    workshops = st.selectbox("Workshops / Certifications", [0, 1, 2, 3], index=1)
    soft_skills = st.slider("Soft Skills Rating", 1.0, 5.0, 3.5, 0.1)

st.markdown('<div class="section-label">Preparation</div>', unsafe_allow_html=True)
c5, c6 = st.columns(2)
with c5:
    extracurricular = st.radio("Extracurricular activities", ["No", "Yes"], horizontal=True, index=1)
with c6:
    placement_training = st.radio("Placement training taken", ["No", "Yes"], horizontal=True, index=1)

st.write("")
predict_clicked = st.button("Assess Placement Chances")


def make_seal_svg(probability: float) -> str:
    """A radial 'wax seal' gauge showing the probability."""
    radius = 70
    circumference = 2 * np.pi * radius
    offset = circumference * (1 - probability)
    color = "#4CAF8C" if probability >= 0.5 else "#D97B5E"
    pct_label = f"{probability * 100:.0f}%"
    return f"""
    <svg viewBox="0 0 168 168" xmlns="http://www.w3.org/2000/svg">
        <circle cx="84" cy="84" r="{radius}" fill="none"
                stroke="rgba(247,243,232,0.14)" stroke-width="10" />
        <circle cx="84" cy="84" r="{radius}" fill="none"
                stroke="{color}" stroke-width="10" stroke-linecap="round"
                stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"
                transform="rotate(-90 84 84)" />
        <circle cx="84" cy="84" r="52" fill="#F7F3E8" />
        <text x="84" y="80" text-anchor="middle" font-family="Fraunces, serif"
              font-weight="700" font-size="28" fill="{color}">{pct_label}</text>
        <text x="84" y="100" text-anchor="middle" font-family="Inter, sans-serif"
              font-size="9" letter-spacing="1.5" fill="#0F1B3C" opacity="0.6">LIKELIHOOD</text>
    </svg>
    """


if predict_clicked:
    row = {
        "StudentID": student_id_fill,
        "CGPA": cgpa,
        "Internships": internships,
        "Projects": projects,
        "Workshops/Certifications": workshops,
        "AptitudeTestScore": aptitude,
        "SoftSkillsRating": soft_skills,
        "ExtracurricularActivities": 1 if extracurricular == "Yes" else 0,
        "PlacementTraining": 1 if placement_training == "Yes" else 0,
        "SSC_Marks": ssc_marks,
        "HSC_Marks": hsc_marks,
    }
    X = np.array([[row[c] for c in feature_cols]])
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]
    placed = prediction == 1

    verdict_text = "Likely Placed" if placed else "Placement At Risk"
    verdict_class = "placed" if placed else "notplaced"
    sub_text = (
        "This profile clears the model's placement threshold."
        if placed
        else "This profile falls short of the model's placement threshold."
    )

    # A real Streamlit container (stable, targetable via .st-key-result_panel)
    # instead of a raw div opened/closed across separate st.markdown calls.
    with st.container(key="result_panel"):
        st.markdown(f'<div class="seal">{make_seal_svg(probability)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="verdict {verdict_class}">{verdict_text}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="verdict-sub">{sub_text}</div>', unsafe_allow_html=True)

        coefs = model.coef_[0]
        contributions = coefs * X_scaled[0]
        display_feats = [
            ("CGPA", contributions[feature_cols.index("CGPA")]),
            ("Internships", contributions[feature_cols.index("Internships")]),
            ("Projects", contributions[feature_cols.index("Projects")]),
            ("Workshops/Certs", contributions[feature_cols.index("Workshops/Certifications")]),
            ("Aptitude Score", contributions[feature_cols.index("AptitudeTestScore")]),
            ("Soft Skills", contributions[feature_cols.index("SoftSkillsRating")]),
            ("Extracurriculars", contributions[feature_cols.index("ExtracurricularActivities")]),
            ("Placement Training", contributions[feature_cols.index("PlacementTraining")]),
            ("SSC Marks", contributions[feature_cols.index("SSC_Marks")]),
            ("HSC Marks", contributions[feature_cols.index("HSC_Marks")]),
        ]
        max_abs = max(abs(v) for _, v in display_feats) or 1.0

        rows_html = ""
        for name, val in sorted(display_feats, key=lambda x: -abs(x[1])):
            pct = abs(val) / max_abs * 100
            bar_color = "#4CAF8C" if val >= 0 else "#D97B5E"
            rows_html += (
                f'<div class="factor-row">'
                f'<span class="fname">{name}</span>'
                f'<span class="fbar-wrap"><span class="fbar" '
                f'style="width:{pct:.0f}%; background:{bar_color};"></span></span>'
                f'<span>{"+" if val >= 0 else ""}{val:.2f}</span>'
                f'</div>'
            )
        st.markdown(f'<div>{rows_html}</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footnote">Trained on historical placement records &middot; '
    "Logistic Regression &middot; for guidance only, not a guarantee</div>",
    unsafe_allow_html=True,
)