import os
import time
import pandas as pd
import joblib
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EV Price AI - Intelligent Electric Vehicle Price Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# MODEL LOADING WITH FALLBACK PATHS
# ============================================================

@st.cache_resource
def load_model():
    """
    Loads the trained machine learning pipeline from ev_price_prediction_model.pkl.
    Supports multiple resolution paths for robust local and cloud deployment.
    """
    model_filename = "ev_price_prediction_model.pkl"
    candidate_paths = [
        model_filename,
        os.path.join(os.path.dirname(__file__), model_filename),
        r"C:\Users\Anuj\Desktop\Projects\EV_Price Predictor Web app\ev_price_prediction_model.pkl"
    ]
    
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception as load_err:
                st.error(f"Error loading model from {path}: {str(load_err)}")
                raise load_err
                
    raise FileNotFoundError(
        f"Model file '{model_filename}' could not be located in candidate paths: {candidate_paths}"
    )

try:
    pipeline = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# ============================================================
# CUSTOM STYLING (DARK GRAPHITE & ELECTRIC CYAN DASHBOARD)
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600;700&display=swap');

:root {
    --bg-surface: #070B12;
    --bg-surface-secondary: #0B111A;
    --card-main: #111720;
    --card-secondary: #151C26;
    --card-accent: #1A222D;
    --cyan-primary: #00D9FF;
    --cyan-bright: #00E5FF;
    --blue-electric: #168BFF;
    --text-primary: #F2F7FA;
    --text-secondary: #8B98A7;
    --text-muted: #657180;
    --border-subtle: rgba(255, 255, 255, 0.07);
    --glow-cyan: rgba(0, 217, 255, 0.15);
}

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Base App Container */
.stApp {
    background-color: var(--bg-surface);
    color: var(--text-primary);
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Top Navigation Bar */
.top-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 24px;
    background-color: var(--card-main);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    margin-bottom: 24px;
}

.top-nav-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.top-nav-icon {
    font-size: 20px;
    color: var(--cyan-primary);
}

.top-nav-title {
    font-weight: 800;
    font-size: 17px;
    letter-spacing: 0.5px;
    color: var(--text-primary);
}

.top-nav-subtitle {
    font-size: 12px;
    color: var(--text-secondary);
    border-left: 1px solid var(--border-subtle);
    padding-left: 12px;
    margin-left: 4px;
}

.top-nav-status {
    display: flex;
    align-items: center;
    gap: 16px;
}

.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    color: var(--cyan-primary);
    background: rgba(0, 217, 255, 0.08);
    border: 1px solid rgba(0, 217, 255, 0.2);
    padding: 5px 12px;
    border-radius: 20px;
}

.status-dot {
    width: 6px;
    height: 6px;
    background-color: var(--cyan-bright);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--cyan-bright);
}

/* Compact Hero Section */
.hero-box {
    background-color: var(--card-main);
    border: 1px solid var(--border-subtle);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 20px;
}

.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    color: var(--cyan-bright);
    background: var(--card-secondary);
    border: 1px solid var(--border-subtle);
    padding: 4px 12px;
    border-radius: 12px;
    margin-bottom: 12px;
}

.hero-heading {
    font-size: 32px;
    font-weight: 800;
    line-height: 1.15;
    color: var(--text-primary);
    margin-bottom: 8px;
}

.hero-highlight {
    color: var(--cyan-bright);
}

.hero-sub {
    font-size: 14px;
    color: var(--text-secondary);
    max-width: 680px;
    line-height: 1.5;
}

/* Model Architecture Horizontal Card */
.arch-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background-color: var(--card-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 16px 24px;
    margin-bottom: 24px;
}

.arch-left {
    display: flex;
    align-items: center;
    gap: 14px;
}

.arch-icon-box {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: var(--card-accent);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--cyan-primary);
    font-size: 18px;
    border: 1px solid var(--border-subtle);
}

.arch-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    color: var(--text-muted);
    text-transform: uppercase;
}

.arch-val {
    font-size: 15px;
    font-weight: 700;
    color: var(--text-primary);
}

.arch-right {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    font-weight: 700;
    color: var(--cyan-primary);
    background: var(--card-accent);
    padding: 6px 14px;
    border-radius: 10px;
    border: 1px solid var(--border-subtle);
}

/* Main Section Header inside Card */
.config-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
}

.config-title-group {
    display: flex;
    align-items: center;
    gap: 10px;
}

.config-title {
    font-size: 20px;
    font-weight: 800;
    color: var(--text-primary);
}

.config-sub {
    font-size: 13px;
    color: var(--text-secondary);
}

.specs-badge {
    text-align: center;
    background: var(--card-accent);
    border: 1px solid var(--border-subtle);
    padding: 6px 14px;
    border-radius: 10px;
}

.specs-num {
    font-size: 16px;
    font-weight: 800;
    color: var(--cyan-bright);
    line-height: 1;
}

.specs-txt {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1px;
    color: var(--text-muted);
}

/* Form Container */
.form-card {
    background-color: var(--card-main);
    border: 1px solid var(--border-subtle);
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 24px;
}

/* Streamlit Tabs Customization */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: var(--bg-surface-secondary);
    padding: 6px;
    border-radius: 14px;
    border: 1px solid var(--border-subtle);
    margin-bottom: 20px;
}

.stTabs [data-baseweb="tab"] {
    height: 42px;
    border-radius: 10px;
    color: var(--text-secondary) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    background: transparent !important;
    border: none !important;
    padding: 0 16px !important;
}

.stTabs [aria-selected="true"] {
    background-color: var(--card-main) !important;
    color: var(--cyan-bright) !important;
    border: 1px solid var(--border-subtle) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
}

/* Custom Input Labels & Fields */
label {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    margin-bottom: 4px !important;
}

.stTextInput input, .stNumberInput input {
    background-color: var(--card-secondary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    padding: 10px 14px !important;
    font-size: 14px !important;
}

.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--cyan-primary) !important;
    box-shadow: 0 0 0 2px var(--glow-cyan) !important;
}

.stSelectbox div[data-baseweb="select"] {
    background-color: var(--card-secondary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

/* Slider Customization */
.stSlider div[data-baseweb="slider"] {
    margin-top: 6px;
}

/* Full Width Primary Predict Button */
div.stButton > button {
    width: 100%;
    height: 52px;
    border: none;
    border-radius: 12px;
    background: linear-gradient(110deg, #00D9FF, #168BFF);
    color: #050911;
    font-weight: 800;
    font-size: 15px;
    letter-spacing: 0.5px;
    cursor: pointer;
    box-shadow: 0 6px 20px var(--glow-cyan);
    transition: all 0.25s ease;
    margin-top: 12px;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 28px rgba(0, 217, 255, 0.35);
    background: linear-gradient(110deg, #00E5FF, #1A92FF);
    color: #000000;
}

/* Prediction Result Card */
.result-card {
    background: linear-gradient(145deg, #111720, #151C26);
    border: 1px solid rgba(0, 217, 255, 0.3);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4), 0 0 25px var(--glow-cyan);
    margin-top: 24px;
    margin-bottom: 24px;
}

.result-header {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    color: var(--cyan-primary);
    text-transform: uppercase;
    margin-bottom: 6px;
}

.result-price-main {
    font-size: clamp(40px, 6vw, 64px);
    font-weight: 900;
    color: var(--text-primary);
    line-height: 1.1;
    margin-bottom: 16px;
}

.result-metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-top: 20px;
}

.res-metric-box {
    background: var(--card-accent);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 14px 10px;
    text-align: center;
}

.res-metric-title {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-bottom: 4px;
}

.res-metric-val {
    font-size: 15px;
    font-weight: 800;
    color: var(--cyan-bright);
}

/* Technical Info Strip */
.tech-card-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}

.tech-card {
    background: var(--card-main);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 16px 20px;
}

.tech-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-bottom: 4px;
}

.tech-val {
    font-size: 14px;
    font-weight: 700;
    color: var(--text-primary);
}

/* Minimal Footer */
.footer-box {
    text-align: center;
    border-top: 1px solid var(--border-subtle);
    padding-top: 24px;
    padding-bottom: 24px;
    margin-top: 32px;
    color: var(--text-muted);
    font-size: 12px;
    line-height: 1.6;
}

@media (max-width: 768px) {
    .hero-heading {
        font-size: 24px;
    }
    .result-metrics-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    .tech-card-grid {
        grid-template-columns: 1fr;
    }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# TOP NAVIGATION BAR
# ============================================================

st.markdown("""
<div class="top-nav">
    <div class="top-nav-brand">
        <span class="top-nav-icon">⚡</span>
        <span class="top-nav-title">EV PRICE AI</span>
        <span class="top-nav-subtitle">Intelligent Electric Vehicle Price Prediction</span>
    </div>
    <div class="top-nav-status">
        <div class="status-indicator">
            <span class="status-dot"></span>
            AI MODEL ONLINE
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# HERO SECTION & MODEL ARCHITECTURE CARD
# ============================================================

st.markdown("""
<div class="hero-box">
    <div class="hero-pill">⚙ RANDOM FOREST ML MODEL</div>
    <div class="hero-heading">
        Predict Your EV's <span class="hero-highlight">True Value</span>
    </div>
    <div class="hero-sub">
        AI-powered electric vehicle price prediction using multi-tree regression algorithms calibrated on verified transaction records.
    </div>
</div>

<div class="arch-card">
    <div class="arch-left">
        <div class="arch-icon-box">⚡</div>
        <div>
            <div class="arch-title">MODEL ARCHITECTURE</div>
            <div class="arch-val">Random Forest Regressor</div>
        </div>
    </div>
    <div class="arch-right">
        ⚙ 21 Features Pipeline
    </div>
</div>
""", unsafe_allow_html=True)

# MODEL STATUS ALERT IF UNABLE TO LOAD
if not model_loaded:
    st.error("⚠️ Model could not be loaded. Please check that ev_price_prediction_model.pkl is present.")
    with st.expander("Technical Diagnostic Details"):
        st.code(model_error)

# ============================================================
# VEHICLE CONFIGURATION STEP-BASED INPUT FORM
# ============================================================

st.markdown("""
<div class="form-card">
    <div class="config-header">
        <div class="config-title-group">
            <span style="color: var(--cyan-primary); font-size: 20px;">⚙</span>
            <div>
                <div class="config-title">Vehicle Configuration</div>
                <div class="config-sub">Configure 21 parameters for accurate valuation</div>
            </div>
        </div>
        <div class="specs-badge">
            <div class="specs-num">21</div>
            <div class="specs-txt">SPECS</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# 4 STEP TABBED INTERFACE FOR COMPACT INFORMATION DENSITY
tab1, tab2, tab3, tab4 = st.tabs([
    "STEP 1/4: Basic Info",
    "STEP 2/4: Performance Telemetry",
    "STEP 3/4: Features & Market",
    "STEP 4/4: Final Config & Predict"
])

# ------------------------------------------------------------
# STEP 1/4: BASIC INFORMATION
# ------------------------------------------------------------
with tab1:
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        brand = st.text_input(
            "EV Brand",
            value="",
            placeholder="e.g. Tesla",
            help="Manufacturer or brand name of the EV."
        )
        model_name = st.text_input(
            "EV Model",
            value="",
            placeholder="e.g. Model 3",
            help="Model name of the vehicle."
        )
        variant = st.text_input(
            "Powertrain Variant",
            value="",
            placeholder="e.g. Long Range",
            help="Trim or variant specification."
        )
    with col2:
        body_type = st.selectbox(
            "Body Architecture / Body Type",
            options=["Sedan", "SUV", "Hatchback", "Coupe", "MPV", "Truck", "Van"],
            index=0,
            help="Vehicle body design category."
        )
        market_segment = st.selectbox(
            "Market Segment",
            options=["Budget", "Mid-range", "Premium", "Luxury"],
            index=2,
            help="Target market classification."
        )
        country_of_origin = st.selectbox(
            "Country of Origin",
            options=["US", "Germany", "China", "Japan", "South Korea"],
            index=0,
            help="Manufacturing country location."
        )

# ------------------------------------------------------------
# STEP 2/4: PERFORMANCE TELEMETRY
# ------------------------------------------------------------
with tab2:
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        battery_capacity = st.slider(
            "Battery Capacity (kWh)",
            min_value=5.0,
            max_value=250.0,
            value=75.0,
            step=1.0,
            help="Usable battery pack capacity in kilowatt-hours."
        )
        range_miles = st.slider(
            "Driving Range (miles)",
            min_value=10.0,
            max_value=1000.0,
            value=350.0,
            step=5.0,
            help="EPA driving range in miles."
        )
        charging_speed_kw = st.slider(
            "Charging Speed (kW)",
            min_value=10.0,
            max_value=500.0,
            value=150.0,
            step=5.0,
            help="Peak DC fast charging speed."
        )
        drive_type = st.selectbox(
            "Drive Type Layout",
            options=["AWD", "FWD", "RWD"],
            index=0,
            help="Drivetrain layout (All-Wheel, Front-Wheel, or Rear-Wheel)."
        )
    with col2:
        horsepower = st.slider(
            "Horsepower (HP)",
            min_value=20.0,
            max_value=2000.0,
            value=300.0,
            step=10.0,
            help="Total motor power output in horsepower."
        )
        torque_nm = st.slider(
            "Torque (Nm)",
            min_value=50.0,
            max_value=2000.0,
            value=450.0,
            step=10.0,
            help="Motor torque in Newton-meters."
        )
        acceleration_0_60_mph = st.slider(
            "0-60 mph Acceleration (sec)",
            min_value=1.0,
            max_value=15.0,
            value=4.5,
            step=0.1,
            help="0 to 60 mph time in seconds."
        )
        top_speed_mph = st.slider(
            "Top Speed (mph)",
            min_value=50.0,
            max_value=300.0,
            value=150.0,
            step=5.0,
            help="Maximum vehicle speed."
        )

# ------------------------------------------------------------
# STEP 3/4: FEATURES & MARKET
# ------------------------------------------------------------
with tab3:
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        weight_kg = st.slider(
            "Curb Weight (kg)",
            min_value=500.0,
            max_value=5000.0,
            value=1800.0,
            step=25.0,
            help="Vehicle curb weight in kilograms."
        )
        cargo_volume_cubic_ft = st.slider(
            "Cargo Volume (cu ft)",
            min_value=5.0,
            max_value=150.0,
            value=35.0,
            step=1.0,
            help="Available cargo space in cubic feet."
        )
        seating_capacity = st.selectbox(
            "Seating Capacity",
            options=[2, 4, 5, 7],
            index=2,
            help="Number of seats."
        )
        year = st.selectbox(
            "Model Year",
            options=[2020, 2021, 2022, 2023, 2024, 2025, 2026],
            index=4,
            help="Manufacturing model year."
        )
    with col2:
        autopilot_level = st.selectbox(
            "Autopilot Level",
            options=[0, 1, 2, 3],
            index=2,
            help="Autonomous driving assistance level."
        )
        safety_rating = st.selectbox(
            "Safety Rating (Stars)",
            options=[3, 4, 5],
            index=2,
            help="NHTSA / Euro NCAP safety rating."
        )
        warranty_years = st.selectbox(
            "Warranty Period (Years)",
            options=[3, 4, 5, 6, 7, 8],
            index=2,
            help="Manufacturer warranty in years."
        )

# ------------------------------------------------------------
# STEP 4/4: FINAL CONFIGURATION & PREDICT TRIGGER
# ------------------------------------------------------------
with tab4:
    st.markdown("""
    <div style="background: var(--card-secondary); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 16px; margin-bottom: 16px;">
        <div style="font-size: 11px; font-weight: 700; color: var(--cyan-primary); letter-spacing: 1px; margin-bottom: 8px;">SPECIFICATION TELEMETRY SUMMARY</div>
        <div style="font-size: 13px; color: var(--text-secondary); line-height: 1.6;">
            Configured 21 model parameters ready for Random Forest valuation. Click the button below to initiate price inference.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    predict_btn = st.button("⚡ PREDICT EV PRICE", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Error details checkbox
debug_mode = st.checkbox("Show technical error details", value=False)

# ============================================================
# PREDICTION EVALUATION & RESULT CARD
# ============================================================

if predict_btn:
    if not brand.strip():
        st.warning("⚠️ Please enter the EV Brand in Step 1.")
    elif not model_name.strip():
        st.warning("⚠️ Please enter the EV Model in Step 1.")
    elif not variant.strip():
        st.warning("⚠️ Please enter the EV Variant in Step 1.")
    elif not model_loaded:
        st.error("❌ Model could not be loaded. Please check that ev_price_prediction_model.pkl is present.")
    else:
        try:
            # Construct DataFrame with EXACT 21 feature schema expected by the trained pipeline
            input_data = pd.DataFrame({
                "brand": [brand.strip()],
                "model": [model_name.strip()],
                "variant": [variant.strip()],
                "battery_capacity_kwh": [float(battery_capacity)],
                "range_miles": [float(range_miles)],
                "horsepower": [float(horsepower)],
                "body_type": [body_type],
                "weight_kg": [float(weight_kg)],
                "charging_speed_kw": [float(charging_speed_kw)],
                "autopilot_level": [int(autopilot_level)],
                "year": [int(year)],
                "top_speed_mph": [float(top_speed_mph)],
                "acceleration_0_60_mph": [float(acceleration_0_60_mph)],
                "drive_type": [drive_type],
                "seating_capacity": [int(seating_capacity)],
                "warranty_years": [int(warranty_years)],
                "market_segment": [market_segment],
                "cargo_volume_cubic_ft": [float(cargo_volume_cubic_ft)],
                "torque_nm": [float(torque_nm)],
                "country_of_origin": [country_of_origin],
                "safety_rating": [int(safety_rating)]
            })

            with st.spinner("Analyzing vehicle specifications..."):
                time.sleep(0.3)
                raw_pred = pipeline.predict(input_data)[0]
                prediction_val = float(raw_pred)

            st.markdown(f"""
            <div class="result-card">
                <div class="result-header">ESTIMATED EV PRICE</div>
                <div class="result-price-main">${prediction_val:,.2f}</div>
                <div style="font-size: 13px; color: var(--text-secondary);">Generated using the trained Random Forest machine learning model</div>
                
                <div class="result-metrics-grid">
                    <div class="res-metric-box">
                        <div class="res-metric-title">Battery</div>
                        <div class="res-metric-val">{battery_capacity:g} kWh</div>
                    </div>
                    <div class="res-metric-box">
                        <div class="res-metric-title">Range</div>
                        <div class="res-metric-val">{range_miles:g} mi</div>
                    </div>
                    <div class="res-metric-box">
                        <div class="res-metric-title">Horsepower</div>
                        <div class="res-metric-val">{horsepower:g} HP</div>
                    </div>
                    <div class="res-metric-box">
                        <div class="res-metric-title">Year</div>
                        <div class="res-metric-val">{year}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as pred_err:
            st.error("Prediction failed. Please verify the vehicle inputs and model configuration.")
            if debug_mode:
                st.code(str(pred_err))

# ============================================================
# TECHNICAL INFORMATION CARDS & FOOTER
# ============================================================

st.markdown("""
<div class="tech-card-grid">
    <div class="tech-card">
        <div class="tech-label">MODEL</div>
        <div class="tech-val">Random Forest</div>
    </div>
    <div class="tech-card">
        <div class="tech-label">FEATURES</div>
        <div class="tech-val">21 Vehicle Specifications</div>
    </div>
    <div class="tech-card">
        <div class="tech-label">PREDICTION</div>
        <div class="tech-val">EV Price Estimation</div>
    </div>
</div>

<div class="footer-box">
    ⚡ <b>EV PRICE AI</b> &nbsp;•&nbsp; Intelligent Electric Vehicle Price Prediction<br>
    Python • Streamlit • Scikit-learn
</div>
""", unsafe_allow_html=True)
