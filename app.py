import streamlit as st
import joblib
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Household Income Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ── Encoding maps ──────────────────────────────────────────────────────────────
EDUCATION_MAP    = {"High School": 0, "Bachelor's": 1, "Master's": 2, "Doctorate": 3}
OCCUPATION_MAP   = {"Professional": 0, "Manager": 1, "Technical": 2, "Sales": 3, "Services": 4, "Retired": 5}
LOCATION_MAP     = {"Urban": 0, "Suburban": 1, "Rural": 2}
MARITAL_MAP      = {"Single": 0, "Married": 1, "Divorced": 2, "Widowed": 3}
EMPLOYMENT_MAP   = {"Full-time": 0, "Part-time": 1, "Self-employed": 2, "Unemployed": 3}
HOMEOWN_MAP      = {"Own": 0, "Rent": 1}
HOUSING_MAP      = {"House": 0, "Apartment": 1, "Condo": 2, "Townhouse": 3}
GENDER_MAP       = {"Male": 0, "Female": 1, "Other": 2}
TRANSPORT_MAP    = {"Car": 0, "Public Transit": 1, "Bicycle": 2, "Walking": 3}

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #080c14 !important;
    font-family: 'DM Sans', sans-serif;
    color: #e2e8f0;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(16, 185, 129, 0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 85% 90%,  rgba(139, 92, 246, 0.10) 0%, transparent 55%),
        #080c14 !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stDecoration"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container { max-width: 1280px !important; padding: 2rem 2.5rem !important; }

/* ── Hero header ── */
.hero-wrap {
    text-align: center;
    padding: 3.5rem 1rem 2.5rem;
    position: relative;
}
.hero-eyebrow {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #10b981;
    background: rgba(16, 185, 129, 0.10);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 999px;
    padding: 0.35rem 1rem;
    margin-bottom: 1.4rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.4rem, 5vw, 3.8rem);
    font-weight: 400;
    line-height: 1.12;
    letter-spacing: -0.02em;
    color: #f0f6ff;
    margin-bottom: 0.9rem;
}
.hero-title em {
    font-style: italic;
    background: linear-gradient(135deg, #10b981 0%, #34d399 50%, #6ee7b7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1rem;
    color: #64748b;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Section labels ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 2.4rem 0 1.2rem;
}
.section-pip {
    width: 4px;
    height: 22px;
    border-radius: 2px;
    background: linear-gradient(180deg, #10b981, #059669);
    flex-shrink: 0;
}
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    color: #cbd5e1;
    letter-spacing: -0.01em;
}
.section-sub {
    font-size: 0.78rem;
    color: #475569;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-left: auto;
}

/* ── Glass card ── */
.glass-card {
    background: rgba(15, 23, 42, 0.70);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 20px;
    padding: 2rem 2.2rem;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    box-shadow:
        0 1px 0 rgba(255,255,255,0.05) inset,
        0 24px 64px rgba(0,0,0,0.45);
    margin-bottom: 1.5rem;
}

/* ── Streamlit widget overrides ── */
label, [data-baseweb="form-control-label"] {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    margin-bottom: 0.3rem !important;
}

/* Select boxes */
[data-baseweb="select"] > div {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-baseweb="select"] > div:hover {
    border-color: rgba(16, 185, 129, 0.35) !important;
}
[data-baseweb="select"] > div:focus-within {
    border-color: rgba(16, 185, 129, 0.6) !important;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.12) !important;
}
[data-baseweb="select"] svg { color: #475569 !important; }
[data-baseweb="popover"] {
    background: #0f172a !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 12px !important;
}
[role="option"]:hover { background: rgba(16, 185, 129, 0.12) !important; }
[aria-selected="true"] { background: rgba(16, 185, 129, 0.18) !important; }

/* Number inputs */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: rgba(16, 185, 129, 0.6) !important;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.12) !important;
    outline: none !important;
}

/* Slider */
[data-testid="stSlider"] {
    padding-top: 0.3rem !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #10b981, #059669) !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: #10b981 !important;
    border: 3px solid #080c14 !important;
    box-shadow: 0 0 0 2px #10b981, 0 4px 12px rgba(16,185,129,0.4) !important;
    width: 18px !important;
    height: 18px !important;
}
[data-testid="stSlider"] [data-testid="stTickBar"] { display: none !important; }

/* Number input stepper buttons */
[data-testid="stNumberInput"] button {
    background: rgba(16, 185, 129, 0.10) !important;
    border-color: rgba(16, 185, 129, 0.20) !important;
    color: #10b981 !important;
    border-radius: 8px !important;
}
[data-testid="stNumberInput"] button:hover {
    background: rgba(16, 185, 129, 0.20) !important;
}

/* ── Predict button ── */
[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%) !important;
    color: #022c22 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.03em !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.9rem 2rem !important;
    cursor: pointer !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow:
        0 1px 0 rgba(255,255,255,0.25) inset,
        0 8px 32px rgba(16, 185, 129, 0.35),
        0 2px 8px rgba(0,0,0,0.4) !important;
    position: relative;
    overflow: hidden;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow:
        0 1px 0 rgba(255,255,255,0.25) inset,
        0 12px 40px rgba(16, 185, 129, 0.50),
        0 4px 16px rgba(0,0,0,0.4) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg,
        rgba(16, 185, 129, 0.08) 0%,
        rgba(5, 150, 105, 0.05) 50%,
        rgba(139, 92, 246, 0.05) 100%);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 24px;
    padding: 3rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow:
        0 0 0 1px rgba(16,185,129,0.08) inset,
        0 32px 80px rgba(0,0,0,0.5),
        0 0 80px rgba(16,185,129,0.06);
    animation: resultReveal 0.6s cubic-bezier(0.22, 1, 0.36, 1) both;
}
@keyframes resultReveal {
    from { opacity: 0; transform: translateY(20px) scale(0.97); }
    to   { opacity: 1; transform: translateY(0)    scale(1); }
}
.result-card::before {
    content: '';
    position: absolute;
    top: -60px; left: 50%; transform: translateX(-50%);
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(16,185,129,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.result-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #10b981;
    margin-bottom: 1rem;
}
.result-amount {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(3rem, 8vw, 5.5rem);
    font-weight: 400;
    letter-spacing: -0.03em;
    line-height: 1;
    color: #f0fdf4;
    text-shadow: 0 0 60px rgba(16,185,129,0.3);
    margin-bottom: 0.5rem;
}
.result-per-year {
    font-size: 0.85rem;
    color: #64748b;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.result-divider {
    width: 60px; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(16,185,129,0.5), transparent);
    margin: 1.2rem auto;
}
.result-sub {
    font-size: 0.82rem;
    color: #475569;
    line-height: 1.5;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3.5rem 2rem;
    border: 1px dashed rgba(255,255,255,0.08);
    border-radius: 24px;
    background: rgba(15, 23, 42, 0.4);
}
.empty-icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.4; }
.empty-text { font-size: 0.9rem; color: #334155; }

/* ── Footer ── */
.footer-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    margin-top: 4rem;
    padding: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    font-size: 0.78rem;
    color: #334155;
}
.footer-bar .credit {
    color: #475569;
    font-weight: 500;
    letter-spacing: 0.02em;
}
.footer-bar .credit span {
    color: #10b981;
    font-weight: 600;
}
.footer-dot { opacity: 0.3; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.05) !important; margin: 0.5rem 0 !important; }

/* ── Slider value display ── */
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
    color: #10b981 !important;
    font-weight: 600 !important;
}

/* Remove default padding on columns */
[data-testid="column"] { padding: 0 0.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">✦ AI-Powered Prediction Engine</div>
    <h1 class="hero-title">Household <em>Income</em><br>Predictor</h1>
    <p class="hero-sub">Enter your household profile below and get an instant, data-driven income estimate powered by gradient boosting.</p>
</div>
""", unsafe_allow_html=True)

# ── Layout: inputs (left 65%) | result (right 35%) ────────────────────────────
col_inputs, col_result = st.columns([13, 7], gap="large")

with col_inputs:
    # ── SECTION 1: Demographics ───────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <div class="section-pip"></div>
        <span class="section-title">Demographics</span>
        <span class="section-sub">Personal Info</span>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            age = st.slider("Age", 18, 100, 35,
                help="Your current age")
        with c2:
            gender = st.selectbox("Gender", list(GENDER_MAP.keys()),
                help="Select your gender identity")
        with c3:
            marital = st.selectbox("Marital Status", list(MARITAL_MAP.keys()),
                help="Current marital status")

        c4, c5 = st.columns(2, gap="medium")
        with c4:
            education = st.selectbox("Education Level", list(EDUCATION_MAP.keys()),
                help="Highest level of education completed")
        with c5:
            dependents = st.number_input("Number of Dependents",
                min_value=0, max_value=10, value=1, step=1,
                help="Total number of financial dependents")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── SECTION 2: Employment & Experience ────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <div class="section-pip" style="background: linear-gradient(180deg, #8b5cf6, #7c3aed);"></div>
        <span class="section-title">Employment & Experience</span>
        <span class="section-sub">Career</span>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        c6, c7 = st.columns(2, gap="medium")
        with c6:
            occupation = st.selectbox("Occupation", list(OCCUPATION_MAP.keys()),
                help="Your primary occupation category")
        with c7:
            employment = st.selectbox("Employment Status", list(EMPLOYMENT_MAP.keys()),
                help="Current employment arrangement")

        work_exp = st.slider("Work Experience (years)", 0, 50, 10,
            help="Total years of professional work experience")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── SECTION 3: Housing & Living ───────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <div class="section-pip" style="background: linear-gradient(180deg, #f59e0b, #d97706);"></div>
        <span class="section-title">Housing & Living Arrangements</span>
        <span class="section-sub">Lifestyle</span>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        c8, c9, c10 = st.columns(3, gap="medium")
        with c8:
            location = st.selectbox("Location", list(LOCATION_MAP.keys()),
                help="Geographic area type")
        with c9:
            homeownership = st.selectbox("Homeownership", list(HOMEOWN_MAP.keys()),
                help="Do you own or rent your home?")
        with c10:
            housing_type = st.selectbox("Type of Housing", list(HOUSING_MAP.keys()),
                help="What kind of home do you live in?")

        c11, c12 = st.columns(2, gap="medium")
        with c11:
            household_size = st.number_input("Household Size",
                min_value=1, max_value=10, value=3, step=1,
                help="Total number of people in the household")
        with c12:
            transport = st.selectbox("Primary Transportation", list(TRANSPORT_MAP.keys()),
                help="Main mode of transport you use")

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Predict button ─────────────────────────────────────────────────────────
    predict_clicked = st.button("✦  Predict Household Income", use_container_width=True)

# ── Result panel ───────────────────────────────────────────────────────────────
with col_result:
    st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

    if predict_clicked:
        features = np.array([[
            age,
            EDUCATION_MAP[education],
            OCCUPATION_MAP[occupation],
            dependents,
            LOCATION_MAP[location],
            work_exp,
            MARITAL_MAP[marital],
            EMPLOYMENT_MAP[employment],
            household_size,
            HOMEOWN_MAP[homeownership],
            HOUSING_MAP[housing_type],
            GENDER_MAP[gender],
            TRANSPORT_MAP[transport],
        ]])

        prediction = model.predict(features)[0]
        monthly = prediction / 12

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Estimated Annual Income</div>
            <div class="result-amount">${prediction:,.0f}</div>
            <div class="result-per-year">per year</div>
            <div class="result-divider"></div>
            <div class="result-sub">
                ≈ &nbsp;<strong style="color:#94a3b8">${monthly:,.0f}</strong>&nbsp; per month<br/>
                <span style="color:#1e293b; font-size:0.75rem; margin-top:0.4rem; display:block">
                    Based on your 13-feature profile
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Breakdown insight chips
        st.markdown(f"""
        <div style="margin-top:1.2rem; display:flex; flex-direction:column; gap:0.6rem;">
            <div style="background:rgba(16,185,129,0.07); border:1px solid rgba(16,185,129,0.15);
                border-radius:12px; padding:0.75rem 1rem; display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.78rem; color:#64748b;">Weekly estimate</span>
                <span style="font-size:0.88rem; font-weight:600; color:#10b981;">${prediction/52:,.0f}</span>
            </div>
            <div style="background:rgba(139,92,246,0.07); border:1px solid rgba(139,92,246,0.15);
                border-radius:12px; padding:0.75rem 1rem; display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.78rem; color:#64748b;">Education factor</span>
                <span style="font-size:0.88rem; font-weight:600; color:#a78bfa;">{education}</span>
            </div>
            <div style="background:rgba(245,158,11,0.07); border:1px solid rgba(245,158,11,0.15);
                border-radius:12px; padding:0.75rem 1rem; display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.78rem; color:#64748b;">Experience</span>
                <span style="font-size:0.88rem; font-weight:600; color:#fbbf24;">{work_exp} years</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">◎</div>
            <p class="empty-text">Fill in your profile details<br/>and click <strong style="color:#475569">Predict Household Income</strong><br/>to see your result here.</p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-bar">
    <span class="credit">Created by <span>Garvit Jaiswal</span></span>
    <span class="footer-dot">·</span>
    <span>Household Income Predictor</span>
    <span class="footer-dot">·</span>
    <span>Gradient Boosting Model</span>
</div>
""", unsafe_allow_html=True)
