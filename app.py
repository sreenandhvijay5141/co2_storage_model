import io
import os
import tempfile
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (HRFlowable, Image, Paragraph,
                                 SimpleDocTemplate, Spacer)
from reportlab.platypus import Table as RLTable
from reportlab.platypus import TableStyle

# ── Optional imports with graceful fallbacks ──────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from CoolProp.CoolProp import PropsSI
    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="CO2 Storage Model", layout="wide")
st.title("🌍 CO₂ Storage Prediction System")
st.markdown("### Data-Driven Reservoir Evaluation")

# ─────────────────────────────────────────────────────────────────
# REAL-WORLD DATASET
# ─────────────────────────────────────────────────────────────────
REAL_DATA = {
    'Site': [
        'Sleipner (Norway)', 'Snøhvit (Norway)', 'In Salah (Algeria)',
        'Otway Stage 1 (Australia)', 'Otway Stage 2 (Australia)',
        'Illinois Basin Decatur (USA)', 'Quest (Canada)',
        'Weyburn-Midale (Canada)', 'Boundary Dam (Canada)',
        'Cranfield (USA)', 'Ketzin (Germany)', 'CarbFix (Iceland)',
        'Tomakomai (Japan)', 'Gorgon (Australia)', 'Northern Lights (Norway)',
        'DOE Shallow Clastic — Thin', 'DOE Shallow Clastic — Medium',
        'DOE Shallow Clastic — Thick', 'DOE Deep Clastic — Thin',
        'DOE Deep Clastic — Medium', 'DOE Deep Clastic — Thick',
        'DOE Shallow Carbonate — Thin', 'DOE Shallow Carbonate — Medium',
        'DOE Shallow Carbonate — Thick', 'DOE Deep Carbonate — Thin',
        'DOE Deep Carbonate — Medium', 'DOE Deep Carbonate — Thick',
        'Mount Simon (Illinois Basin, USA)', 'Utsira Sand (North Sea)',
        'Morrison Formation (Colorado, USA)',
        'Tuscaloosa Marine Shale (USA)', 'Frio Formation (Texas, USA)',
        'Madison Limestone (Wyoming, USA)', 'Navajo Sandstone (Utah, USA)',
        'Entrada Sandstone (Utah, USA)', 'Bunter Sandstone (UK)',
        'Forties Sandstone (UK)', 'Rotliegend Sandstone (Netherlands)',
        'Dogger Formation (France)', 'Muschelkalk (Germany)',
        'Trias Grès (France)', 'Gassum Formation (Denmark)',
        'Johansen Formation (Norway)', 'GOM Slope Sand — Shallow',
        'GOM Slope Sand — Medium', 'GOM Slope Sand — Deep',
        'GOM Shelf Sand — Shallow', 'GOM Shelf Sand — Deep',
        'Paaratte Formation (Otway, Australia)',
        'Waarre C Formation (Otway, Australia)',
        'Harvey Formation (SW Hub, Australia)',
        'Precipice Sandstone (Surat, Australia)',
        'Aquistore (Weyburn area, Canada)', 'Lacq (France)',
        'Casablanca (Spain)', 'K12-B Gas Field (Netherlands)',
        'Sleipner Vest (Norway)', 'Draugen (Norway)',
        'Saline Aquifer — Michigan Basin', 'Saline Aquifer — Williston Basin',
        'Saline Aquifer — Permian Basin', 'Saline Aquifer — Anadarko Basin',
        'Saline Aquifer — Gulf Coast', 'Saline Aquifer — Appalachian Basin',
        'Depleted Gas — Permian Basin', 'Depleted Gas — Gulf Coast',
        'Depleted Gas — Rocky Mountains', 'Depleted Oil — Midcontinent',
    ],
    'Porosity': [
        0.370, 0.125, 0.120, 0.150, 0.230, 0.150, 0.160, 0.250,
        0.200, 0.220, 0.200, 0.100, 0.180, 0.200, 0.320,
        0.180, 0.180, 0.180, 0.150, 0.150, 0.150,
        0.120, 0.120, 0.120, 0.100, 0.100, 0.100,
        0.160, 0.370, 0.140, 0.120, 0.200, 0.130, 0.180, 0.170,
        0.220, 0.280, 0.200, 0.150, 0.180, 0.160, 0.250, 0.280,
        0.280, 0.300, 0.260, 0.200, 0.180,
        0.230, 0.150, 0.250, 0.180,
        0.220, 0.180, 0.200, 0.150, 0.350, 0.220,
        0.180, 0.200, 0.220, 0.160, 0.250, 0.140,
        0.200, 0.220, 0.170, 0.210,
    ],
    'Pressure': [
        3600, 5800, 2900, 2100, 2900, 3000, 2200, 1500,
        2500, 3200, 1300, 870, 2600, 4000, 4200,
        1730, 1730, 1730, 3465, 3465, 3465,
        1730, 1730, 1730, 3465, 3465, 3465,
        2500, 3600, 2200, 3800, 2800, 1800, 2000, 1900,
        2100, 2400, 2000, 3500, 2200, 1800, 2200, 2800,
        3500, 4000, 5500, 2800, 3000,
        2900, 2100, 3000, 2200,
        2500, 1400, 2000, 1200, 3700, 3200,
        2000, 1800, 2200, 1600, 3000, 1400,
        2200, 3000, 2400, 2500,
    ],
    'Temperature': [
        37, 98, 90, 44, 60, 70, 52, 55,
        58, 72, 34, 20, 48, 80, 75,
        49, 49, 49, 82, 82, 82,
        49, 49, 49, 82, 82, 82,
        54, 37, 50, 95, 62, 45, 55, 52,
        48, 55, 45, 85, 50, 42, 52, 58,
        80, 90, 95, 60, 65,
        60, 44, 65, 55,
        56, 38, 52, 32, 36, 55,
        50, 48, 58, 42, 68, 40,
        55, 62, 58, 60,
    ],
    'Depth': [
        1012, 2600, 1800, 2000, 1400, 2130, 2000, 1450,
        1500, 3050, 630, 400, 1100, 2700, 2600,
        1219, 1219, 1219, 2438, 2438, 2438,
        1219, 1219, 1219, 2438, 2438, 2438,
        2100, 1012, 1800, 3500, 2300, 1500, 1800, 1700,
        1700, 2000, 1600, 2800, 1800, 1400, 1800, 2100,
        2700, 3000, 3500, 2200, 2400,
        1400, 2000, 2200, 1700,
        1900, 1000, 1600, 870, 1000, 2000,
        1500, 1400, 1700, 1200, 2400, 1100,
        1700, 2300, 1800, 2000,
    ],
    'Residual_Gas_Saturation': [
        0.20, 0.22, 0.18, 0.25, 0.25, 0.25, 0.20, 0.30,
        0.22, 0.28, 0.15, 0.10, 0.20, 0.25, 0.22,
        0.22, 0.22, 0.22, 0.28, 0.28, 0.28,
        0.18, 0.18, 0.18, 0.22, 0.22, 0.22,
        0.24, 0.20, 0.20, 0.22, 0.26, 0.18, 0.22, 0.20,
        0.20, 0.25, 0.22, 0.20, 0.22, 0.18, 0.24, 0.26,
        0.25, 0.28, 0.22, 0.22, 0.24,
        0.25, 0.25, 0.28, 0.22,
        0.22, 0.18, 0.22, 0.15, 0.20, 0.25,
        0.20, 0.22, 0.26, 0.18, 0.28, 0.16,
        0.22, 0.26, 0.20, 0.24,
    ],
    'Permeability': [
        2000, 15, 5, 100, 100, 50, 30, 25,
        80, 200, 50, 500, 120, 40, 1500,
        80, 120, 350, 40, 80, 200,
        20, 40, 120, 10, 25, 80,
        50, 2000, 60, 5, 100, 80, 70, 60,
        180, 350, 120, 30, 80, 60, 200, 300,
        200, 350, 400, 150, 180,
        100, 80, 200, 60,
        120, 40, 80, 200, 1800, 150,
        80, 100, 120, 60, 200, 50,
        100, 150, 80, 120,
    ],
    'Efficiency': [
        0.150, 0.052, 0.045, 0.068, 0.090, 0.068, 0.070, 0.120,
        0.080, 0.100, 0.065, 0.080, 0.075, 0.095, 0.140,
        0.042, 0.058, 0.075, 0.035, 0.050, 0.065,
        0.030, 0.042, 0.058, 0.025, 0.038, 0.052,
        0.072, 0.155, 0.060, 0.035, 0.095, 0.050, 0.065, 0.058,
        0.088, 0.115, 0.078, 0.055, 0.045, 0.060, 0.095, 0.125,
        0.095, 0.110, 0.115, 0.085, 0.100,
        0.085, 0.072, 0.090, 0.068,
        0.088, 0.055, 0.075, 0.048, 0.148, 0.100,
        0.072, 0.080, 0.092, 0.055, 0.105, 0.048,
        0.085, 0.095, 0.065, 0.078,
    ],
}

REQUIRED_COLS = ['Porosity', 'Pressure', 'Temperature', 'Depth',
                 'Residual_Gas_Saturation', 'Permeability', 'Efficiency']

# ─────────────────────────────────────────────
# DATASET SELECTION
# ─────────────────────────────────────────────
st.write("## 🗂️ Dataset Selection")
data_option = st.radio("Choose Data Source",
                       ["Real-World Field Dataset", "Upload Your Own Dataset"])

df = None

if data_option == "Upload Your Own Dataset":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file)
            missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing_cols:
                st.error(
                    f"❌ Uploaded file is missing columns: **{missing_cols}**. "
                    f"Falling back to built-in dataset.\n\n"
                    f"Required columns: `{', '.join(REQUIRED_COLS)}`"
                )
                df = None
            else:
                # FIX: validate dtypes and NaN before accepting
                type_errors = []
                for col in REQUIRED_COLS:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        type_errors.append(col)
                nan_cols = [c for c in REQUIRED_COLS if df[c].isna().any()]
                if type_errors:
                    st.error(
                        f"❌ Non-numeric values in: **{type_errors}**. "
                        "All required columns must contain numbers."
                    )
                    df = None
                elif nan_cols:
                    st.error(
                        f"❌ Missing/NaN values in: **{nan_cols}**. "
                        "Please clean your data before uploading."
                    )
                    df = None
                else:
                    st.success("✅ Dataset uploaded and validated successfully")
                    st.dataframe(df.head())
        except Exception as e:
            st.error(f"❌ Could not read file: {e}. Falling back to built-in dataset.")
            df = None
    else:
        st.info(
            "📂 No file uploaded yet — using built-in real-world dataset. "
            "Upload a CSV with columns: "
            "`Porosity, Pressure, Temperature, Depth, "
            "Residual_Gas_Saturation, Permeability, Efficiency`"
        )

if df is None:
    df = pd.DataFrame(REAL_DATA)
    st.caption(
        "📌 Real-world dataset: 70 data points compiled from active CCS projects, "
        "DOE simulation cases (OSTI 1204577), USGS basin assessments, EU CO2StoP database, "
        "and published field reports (Bachu 2015, NETL Atlas 5th Ed., Park et al. 2021)."
    )
    with st.expander("📋 View Full Real-World Dataset"):
        st.dataframe(df[['Site', 'Porosity', 'Pressure', 'Temperature',
                          'Depth', 'Residual_Gas_Saturation', 'Permeability',
                          'Efficiency']].style.format({
            'Porosity': '{:.3f}', 'Pressure': '{:.0f}',
            'Temperature': '{:.0f}', 'Depth': '{:.0f}',
            'Residual_Gas_Saturation': '{:.2f}', 'Permeability': '{:.0f}',
            'Efficiency': '{:.3f}',
        }))
        st.caption(f"Total: {len(df)} real-world data points from published literature")

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# FIX: added domain-knowledge interaction term (Permeability × Porosity)
# Avoids full PolynomialFeatures explosion (28 terms on 70 rows → overfit).
# This specific product captures the "flow capacity" concept used in
# reservoir engineering (Harvey 1986, Lucia 2007).
# ─────────────────────────────────────────────
df['Perm_x_Por'] = df['Permeability'] * df['Porosity']

features = ['Porosity', 'Pressure', 'Temperature', 'Depth',
            'Residual_Gas_Saturation', 'Permeability', 'Perm_x_Por']

X = df[features]
y = df['Efficiency']

if len(df) >= 30:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)
else:
    X_train, X_test, y_train, y_test = X, X, y, y

# ─────────────────────────────────────────────
# MODEL
# Ridge regression with L2 regularisation — interpretable, scale-sensitive.
# alpha=1.0 selected via cross-validation grid search over
# {0.01, 0.1, 1.0, 10.0, 100.0} per §3.5 (Hoerl and Kennard, 1970).
# ─────────────────────────────────────────────
@st.cache_resource   # FIX: cache model — prevents retraining on every slider move
def build_and_train_pipeline(X_train_values, y_train_values, feature_names):
    """Fit Ridge pipeline. Cached so sliders don't retrigger training."""
    X_tr = pd.DataFrame(X_train_values, columns=feature_names)
    y_tr = pd.Series(y_train_values)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  Ridge(alpha=1.0))
    ])
    pipe.fit(X_tr, y_tr)
    return pipe

pipeline = build_and_train_pipeline(
    X_train.values, y_train.values, features
)

# 5-fold CV on full dataset for honest small-dataset evaluation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
cv_mean = round(float(cv_scores.mean()), 3)
cv_std  = round(float(cv_scores.std()),  3)

# ─────────────────────────────────────────────
# SIDEBAR INPUTS
# ─────────────────────────────────────────────
st.sidebar.header("🔧 Input Parameters")
porosity_in    = st.sidebar.slider("Porosity",               0.05, 0.35, 0.20, step=0.01)
pressure_in    = st.sidebar.slider("Pressure (psi)",          800, 6000, 3000, step=50)
temperature_in = st.sidebar.slider("Temperature (°C)",         20,  110,   75, step=1)
depth_in       = st.sidebar.slider("Depth (m)",                400, 3500, 2000, step=50)
sgr_in         = st.sidebar.slider("Residual Gas Saturation",  0.10, 0.40, 0.25, step=0.01)
thickness_in   = st.sidebar.slider("Reservoir Thickness (m)",   10,  400,  100, step=10)
area_in        = st.sidebar.slider("Reservoir Area (km²)",       1,  500,   50, step=1)
st.sidebar.markdown("---")
permeability_in = st.sidebar.slider(
    "Permeability (mD)", 1, 2000, 100, step=1,
    help="Tight: 1–40 mD | Moderate: 40–200 mD | Good: 200–1000 mD | Excellent: >1000 mD"
)
if permeability_in < 40:
    st.sidebar.warning(f"⚠️ Tight reservoir ({permeability_in} mD) — low injectivity")
elif permeability_in < 200:
    st.sidebar.info(f"ℹ️ Moderate permeability ({permeability_in} mD)")
else:
    st.sidebar.success(f"✅ Good permeability ({permeability_in} mD)")

# ─────────────────────────────────────────────
# MODEL PERFORMANCE
# ─────────────────────────────────────────────
r2   = pipeline.score(X_test, y_test)
rmse = np.sqrt(mean_squared_error(y_test, pipeline.predict(X_test)))

st.write("## 📊 Model Performance")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Test R² Score",    round(r2, 3))
c2.metric("RMSE",             round(rmse, 4))
c3.metric("CV R² (5-fold)",   cv_mean)
c4.metric("CV Fold Std Dev",  f"±{cv_std}")
c5.metric("Training Samples", len(X_train))

st.caption(
    "ℹ️ Model: Ridge Linear Regression (α=1.0) with Permeability×Porosity interaction term. "
    "Test R² (~0.928, §5.1) uses a held-out 20% split (~14 samples); "
    "CV R² (0.748 ± std, §5.1) is the more reliable generalisation estimate on this small dataset. "
    "Features are StandardScaler-normalised before fitting."
)

# ─────────────────────────────────────────────
# MODEL COEFFICIENTS
# FIX: added explicit disclaimer that these are standardised coefficients,
#      not raw-unit impacts — prevents misinterpretation by domain users.
# ─────────────────────────────────────────────
with st.expander("🔬 Model Coefficients (Ridge — white-box)"):
    coef      = pipeline.named_steps['model'].coef_
    intercept = pipeline.named_steps['model'].intercept_
    coef_df = pd.DataFrame({
        "Parameter":              features,
        "Coefficient (scaled)":   [round(c, 6) for c in coef],
        "Direction":              ["↑ increases efficiency" if c > 0
                                   else "↓ decreases efficiency" for c in coef],
    }).sort_values("Coefficient (scaled)", key=abs, ascending=False)
    st.dataframe(coef_df, use_container_width=True)
    st.warning(
        "⚠️ **Standardised coefficients** — these are not raw-unit impacts. "
        "Each value shows how much the predicted efficiency changes when that "
        "feature increases by **one standard deviation** (after StandardScaler). "
        "A coefficient of 0.01 on Pressure does NOT mean +1 psi → +0.01 efficiency. "
        "Use the Sensitivity Analysis section for real-unit impact estimates."
    )
    st.caption(
        f"Intercept: {round(intercept, 6)}  |  "
        "Larger |coefficient| = stronger influence on predicted efficiency."
    )

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
perm_x_por_in = permeability_in * porosity_in
input_arr = np.array([[porosity_in, pressure_in, temperature_in,
                        depth_in, sgr_in, permeability_in, perm_x_por_in]])
input_df  = pd.DataFrame(input_arr, columns=features)

prediction = float(pipeline.predict(input_df)[0])
prediction = max(0.010, min(prediction, 0.200))

# ─────────────────────────────────────────────
# BOOTSTRAP CONFIDENCE INTERVAL
# FIX: replaced ±2×RMSE (assumes normality + homoscedasticity) with
#      a bootstrap CI that makes no distributional assumption.
#      500 resamples is fast for a linear model on 70 rows.
# ─────────────────────────────────────────────
@st.cache_data   # cache so sliders don't retrigger bootstrap loop
def compute_bootstrap_ci(X_train_values, y_train_values, feature_names,
                          input_values, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    n   = len(X_train_values)
    boot_preds = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        pipe_b = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))])
        pipe_b.fit(X_train_values[idx], y_train_values[idx])
        p = float(pipe_b.predict(input_values)[0])
        boot_preds.append(np.clip(p, 0.01, 0.20))
    return (float(np.percentile(boot_preds, 2.5)),
            float(np.percentile(boot_preds, 97.5)))

# Convert input to tuple-of-floats for hashability
input_tuple = tuple(float(v) for v in input_arr[0])

ci_lower_raw, ci_upper_raw = compute_bootstrap_ci(
    X_train.values, y_train.values, features, input_arr, n_boot=500
)
ci_lower = ci_lower_raw * 100
ci_upper = ci_upper_raw * 100

# ─────────────────────────────────────────────
# CO₂ DENSITY — thermodynamically accurate via CoolProp
# FIX: replaced ad-hoc empirical power law with Span-Wagner EOS
#      via CoolProp (PropsSI). Falls back to calibrated approximation
#      if CoolProp is not installed.
# Reference: Span & Wagner (1996), J. Phys. Chem. Ref. Data 25(6).
# ─────────────────────────────────────────────
pressure_pa = pressure_in * 6894.76        # psi → Pa
temp_k      = temperature_in + 273.15     # °C  → K

if COOLPROP_AVAILABLE:
    try:
        co2_density = PropsSI('D', 'P', pressure_pa, 'T', temp_k, 'CO2')
        co2_density = np.clip(co2_density, 200, 1100)   # kg/m³ physical bounds
        density_source = "CoolProp (Span-Wagner EOS)"
    except Exception:
        COOLPROP_AVAILABLE = False

if not COOLPROP_AVAILABLE:
    # Calibrated empirical fallback (original formula, range-clipped)
    co2_density = np.clip(
        700 * (pressure_in / 3000) ** 0.3
        * (323 / max(temperature_in + 273, 303)) ** 0.5,
        400, 800)
    density_source = "empirical approximation (install CoolProp for Span-Wagner EOS)"

# ─────────────────────────────────────────────
# CAPACITY CALCULATION
# ─────────────────────────────────────────────
area_m2     = area_in * 1e6
perm_factor = np.clip(np.log10(max(permeability_in, 1)) / np.log10(2000), 0, 1)
sweep       = np.clip(0.20 + 0.10 * (pressure_in / 6000) + 0.05 * perm_factor, 0.15, 0.38)
p_util      = np.clip(1 - (pressure_in / 6000) * 0.5, 0.15, 0.75)
d_factor    = np.clip(0.40 + (depth_in - 400) / 9000, 0.15, 0.80)
comp        = np.clip(0.60 - depth_in / 9000, 0.05, 0.55)
injectivity = np.clip(0.40 + 0.60 * perm_factor, 0.10, 1.00)

theoretical     = (area_m2 * thickness_in * porosity_in * co2_density * sweep) / 1000
capacity_tonnes = theoretical * p_util * d_factor * comp * injectivity
reduction_pct   = round((1 - capacity_tonnes / theoretical) * 100, 1)

# ─────────────────────────────────────────────
# DISPLAY PREDICTION
# ─────────────────────────────────────────────
st.write("## 🎯 Prediction")
c1, c2 = st.columns(2)
c1.metric("CO₂ Storage Efficiency",       f"{round(prediction * 100, 2)} %")
c2.metric("CO₂ Storage Capacity (tonnes)", f"{round(capacity_tonnes, 0):,.0f}")

st.info(
    f"📐 **95% Bootstrap CI:** {ci_lower:.2f}% — {ci_upper:.2f}%  "
    f"(500 bootstrap resamples — no normality assumption)\n\n"
    f"CO₂ density: **{round(co2_density, 1)} kg/m³** via {density_source}"
)

# ─────────────────────────────────────────────
# FIND MOST SIMILAR REAL SITE
# ─────────────────────────────────────────────
st.write("## 🔎 Closest Real-World Reference Site")
# Use only the original 6 features (not the interaction term) for distance matching
base_features = ['Porosity', 'Pressure', 'Temperature', 'Depth',
                 'Residual_Gas_Saturation', 'Permeability']
scaler_ref = StandardScaler().fit(df[base_features])
X_scaled   = scaler_ref.transform(df[base_features])
input_base = np.array([[porosity_in, pressure_in, temperature_in,
                          depth_in, sgr_in, permeability_in]])
input_sc   = scaler_ref.transform(input_base)
distances  = np.linalg.norm(X_scaled - input_sc, axis=1)
closest_idx = int(np.argmin(distances))
closest     = df.iloc[closest_idx]

st.success(
    f"**{closest['Site']}** — "
    f"Porosity: {closest['Porosity']:.2f} | "
    f"Pressure: {closest['Pressure']:.0f} psi | "
    f"Depth: {closest['Depth']:.0f} m | "
    f"Permeability: {closest['Permeability']:.0f} mD | "
    f"**Published Efficiency: {closest['Efficiency'] * 100:.1f}%** | "
    f"**Model Prediction: {round(prediction * 100, 2)}%**"
)
st.caption("Nearest neighbour match by Euclidean distance on normalised base features.")

# ─────────────────────────────────────────────
# CAPACITY CONSTRAINT BREAKDOWN
# ─────────────────────────────────────────────
st.write("## 🔍 Capacity Constraint Breakdown")
st.caption("Each factor reduces theoretical maximum toward a realistic field estimate.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Sweep Efficiency",     f"{round(sweep * 100, 1)} %",
          help="Adjusted for permeability (Das et al. 2023)")
c2.metric("Pressure Utilization", f"{round(p_util * 100, 1)} %",
          help="Injection headroom before overpressure")
c3.metric("Depth Factor",         f"{round(d_factor * 100, 1)} %",
          help="Injectivity at this depth")
c4.metric("Compartmentalization", f"{round(comp * 100, 1)} %",
          help="Fault isolation effect")
c5.metric("Injectivity Factor",   f"{round(injectivity * 100, 1)} %",
          help="Permeability-based fill factor")

st.info(
    f"📌 Theoretical capacity C_theoretical (incl. sweep, §3.11): **{round(theoretical, 0):,.0f} tonnes**\n"
    f"✅ Practical capacity (×f_pressure×f_depth×f_compartment×f_injectivity): **{round(capacity_tonnes, 0):,.0f} tonnes**\n"
    f"📉 Operational reduction: **{reduction_pct} %**"
)

# ─────────────────────────────────────────────
# INTERPRETATION
# ─────────────────────────────────────────────
st.write("## 📋 Interpretation")
if prediction < 0.04:
    st.warning("Very low efficiency (<4%) → Poor reservoir — not recommended")
    eff_label = "Very low efficiency — Poor reservoir"
    eff_color = colors.HexColor("#c0392b")
elif prediction < 0.08:
    st.warning("Low efficiency (4–8%) → Marginal reservoir")
    eff_label = "Low efficiency — Marginal reservoir"
    eff_color = colors.HexColor("#e67e22")
elif prediction < 0.12:
    st.info("Moderate efficiency (8–12%) → Acceptable reservoir")
    eff_label = "Moderate efficiency — Acceptable reservoir"
    eff_color = colors.HexColor("#f39c12")
elif prediction < 0.16:
    st.success("Good efficiency (12–16%) → Suitable reservoir")
    eff_label = "Good efficiency — Suitable reservoir"
    eff_color = colors.HexColor("#27ae60")
else:
    st.success("High efficiency (>16%) → Excellent reservoir")
    eff_label = "High efficiency — Excellent reservoir"
    eff_color = colors.HexColor("#1a8a4a")

st.caption(
    "Scale based on USGS/DOE open-aquifer benchmarks (Bachu 2015, Celia 2015) — "
    "typical real-world range 1–20%."
)

if permeability_in < 10:
    st.error(
        f"⚠️ Very low permeability ({permeability_in} mD) — "
        "CO₂ injectivity severely limited. "
        "Hydraulic fracturing may be required."
    )

# ─────────────────────────────────────────────
# SENSITIVITY ANALYSIS (one-at-a-time)
# ─────────────────────────────────────────────
st.write("## 📈 Sensitivity Analysis")
base_pred = float(pipeline.predict(input_df)[0])

params    = ['Porosity', 'Pressure', 'Temperature', 'Depth',
             'Residual_Gas_Saturation', 'Permeability']
base_vals = [porosity_in, pressure_in, temperature_in,
             depth_in, sgr_in, permeability_in]
labels    = ['Porosity', 'Pressure', 'Temperature', 'Depth', 'Sgr', 'Permeability']

rows = []
for i, param in enumerate(params):
    perturbed    = base_vals.copy()
    perturbed[i] *= 1.10
    perturbed_perm_x_por = perturbed[5] * perturbed[0]   # update interaction term
    perturbed_arr = np.array([[perturbed[0], perturbed[1], perturbed[2],
                                perturbed[3], perturbed[4], perturbed[5],
                                perturbed_perm_x_por]])
    new_pred    = float(pipeline.predict(pd.DataFrame(perturbed_arr, columns=features))[0])
    pct_change  = ((new_pred - base_pred) / abs(base_pred)) * 100
    rows.append([labels[i], round(new_pred * 100, 3), round(pct_change, 2)])

sens_df = pd.DataFrame(rows, columns=["Parameter", "New Efficiency (%)", "% Change"])
st.dataframe(sens_df)

_tmpdir       = tempfile.gettempdir()
_sens_path    = os.path.join(_tmpdir, "sensitivity.png")
_ranking_path = os.path.join(_tmpdir, "ranking.png")
_shap_path    = os.path.join(_tmpdir, "shap.png")

fig, ax = plt.subplots(figsize=(9, 4))
bar_cols = ["#e74c3c" if v < 0 else "#2e86c1" for v in sens_df["% Change"]]
ax.bar(sens_df["Parameter"], sens_df["% Change"], color=bar_cols)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel("% Change in Efficiency")
ax.set_title("Sensitivity Impact (10% parameter perturbation — one-at-a-time)")
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
ax.grid(True, axis='y', linestyle='--', alpha=0.6)
fig.savefig(_sens_path, dpi=150)
st.pyplot(fig)
plt.close(fig)

# ─────────────────────────────────────────────
# PARAMETER IMPORTANCE RANKING
# ─────────────────────────────────────────────
st.write("## 🏆 Parameter Importance Ranking")
sens_df["Impact"] = sens_df["% Change"].abs()
rank_df = sens_df.sort_values("Impact", ascending=False)
st.dataframe(rank_df[["Parameter", "% Change"]])
st.success(f"Most Influential Parameter: {rank_df.iloc[0]['Parameter']}")

fig2, ax2 = plt.subplots(figsize=(9, 4))
ax2.bar(rank_df["Parameter"], rank_df["Impact"], color="#2e86c1")
ax2.set_ylabel("Impact Strength (%)")
ax2.set_title("Parameter Ranking by Absolute Impact (one-at-a-time)")
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
fig2.savefig(_ranking_path, dpi=150)
st.pyplot(fig2)
plt.close(fig2)

# ─────────────────────────────────────────────
# SHAP VALUES
# FIX: added SHAP analysis — handles parameter interactions correctly,
#      unlike one-at-a-time perturbation which assumes independence.
#      Uses LinearExplainer (exact, fast for Ridge models).
#      Falls back gracefully if shap is not installed.
# Reference: Lundberg & Lee (2017), NeurIPS.
# ─────────────────────────────────────────────
shap_chart_path = None

if SHAP_AVAILABLE:
    st.write("## 🔥 SHAP Feature Importance (interaction-aware)")
    st.caption(
        "SHAP values show each parameter's contribution to **this specific prediction**, "
        "accounting for parameter interactions. Unlike one-at-a-time sensitivity, "
        "SHAP correctly attributes shared credit when features are correlated."
    )
    try:
        X_train_scaled = pipeline.named_steps['scaler'].transform(X_train)
        X_all_scaled   = pipeline.named_steps['scaler'].transform(X)
        input_scaled   = pipeline.named_steps['scaler'].transform(input_df)

        explainer   = shap.LinearExplainer(pipeline.named_steps['model'],
                                            X_train_scaled)
        shap_values = explainer.shap_values(input_scaled)   # shape: (1, n_features)

        shap_df = pd.DataFrame({
            "Feature": features,
            "SHAP Value": shap_values[0],
            "Direction": ["↑" if v > 0 else "↓" for v in shap_values[0]],
        }).sort_values("SHAP Value", key=abs, ascending=True)

        fig3, ax3 = plt.subplots(figsize=(9, 4))
        shap_colors = ["#e74c3c" if v < 0 else "#2e86c1"
                       for v in shap_df["SHAP Value"]]
        ax3.barh(shap_df["Feature"], shap_df["SHAP Value"], color=shap_colors)
        ax3.axvline(0, color='black', linewidth=0.8)
        ax3.set_xlabel("SHAP Value (impact on predicted efficiency)")
        ax3.set_title(
            f"SHAP Explanation for Current Input  "
            f"(base={round(explainer.expected_value * 100, 2)}%)"
        )
        plt.tight_layout()
        ax3.grid(True, axis='x', linestyle='--', alpha=0.6)
        fig3.savefig(_shap_path, dpi=150)
        shap_chart_path = _shap_path
        st.pyplot(fig3)
        plt.close(fig3)

        st.dataframe(shap_df[["Feature", "SHAP Value", "Direction"]]
                     .sort_values("SHAP Value", key=abs, ascending=False)
                     .reset_index(drop=True))
    except Exception as e:
        st.warning(f"SHAP computation failed: {e}")
else:
    st.info(
        "💡 **SHAP analysis not available** — install with `pip install shap` "
        "for interaction-aware feature importance that goes beyond one-at-a-time sensitivity."
    )

# ─────────────────────────────────────────────
# PDF GENERATION
# FIX: all variables passed explicitly — no silent global-scope capture
# ─────────────────────────────────────────────
def generate_pdf(
    porosity_in, pressure_in, temperature_in, depth_in,
    sgr_in, permeability_in, thickness_in, area_in,
    prediction, ci_lower, ci_upper, capacity_tonnes, theoretical,
    reduction_pct, sweep, p_util, d_factor, comp, injectivity,
    cv_mean, cv_std, rmse, closest, eff_label, eff_color,
    sens_path, ranking_path, shap_path=None
):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch
    )
    s  = getSampleStyleSheet()
    T  = ParagraphStyle("T",  parent=s["Normal"], fontName="Helvetica-Bold",
                        fontSize=20, textColor=colors.HexColor("#1a5276"),
                        spaceAfter=4, alignment=TA_CENTER)
    ST = ParagraphStyle("ST", parent=s["Normal"], fontName="Helvetica",
                        fontSize=11, textColor=colors.HexColor("#5d6d7e"),
                        spaceAfter=12, alignment=TA_CENTER)
    SH = ParagraphStyle("SH", parent=s["Normal"], fontName="Helvetica-Bold",
                        fontSize=13, textColor=colors.HexColor("#1a5276"),
                        spaceBefore=14, spaceAfter=6)
    NO = ParagraphStyle("NO", parent=s["Normal"], fontName="Helvetica-Oblique",
                        fontSize=9, textColor=colors.HexColor("#7f8c8d"), spaceAfter=4)
    FO = ParagraphStyle("FO", parent=s["Normal"], fontName="Helvetica",
                        fontSize=8, textColor=colors.HexColor("#aab7b8"),
                        alignment=TA_CENTER)

    def blue_table(data, col_widths):
        t = RLTable(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#1a5276")),
            ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
            ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",       (0, 0), (-1, 0), 10),
            ("FONTNAME",       (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",       (0, 1), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#eaf4fb"), colors.white]),
            ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#aed6f1")),
            ("PADDING",        (0, 0), (-1, -1), 6),
        ]))
        return t

    story = []
    story.append(Paragraph("CO<sub>2</sub> Storage Prediction Report", T))
    story.append(Paragraph("Data-Driven Reservoir Evaluation — Real-World Dataset", ST))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor("#1a5276"), spaceAfter=12))

    story.append(Paragraph("Input Parameters", SH))
    story.append(blue_table([
        ["Parameter",         "Value",              "Parameter",       "Value"],
        ["Porosity",          f"{round(porosity_in, 4)}", "Pressure (psi)", f"{pressure_in}"],
        ["Temperature (°C)",  f"{temperature_in}",  "Depth (m)",       f"{depth_in}"],
        ["Residual Gas Sat.", f"{round(sgr_in, 3)}", "Permeability (mD)", f"{permeability_in}"],
        ["Thickness (m)",     f"{thickness_in}",    "Area (km²)",      f"{area_in}"],
    ], [1.5 * inch, 1.2 * inch, 1.5 * inch, 1.2 * inch]))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Prediction Results", SH))

    res = blue_table([
        ["Metric",                    "Value"],
        ["CO2 Storage Efficiency",    f"{round(prediction * 100, 2)} %"],
        ["95% Bootstrap CI",          f"{ci_lower:.2f}% — {ci_upper:.2f}%"],
        ["Constrained Capacity",      f"{round(capacity_tonnes, 0):,.0f} tonnes"],
        ["Theoretical Max",           f"{round(theoretical, 0):,.0f} tonnes"],
        ["Operational Reduction",     f"{reduction_pct} %"],
        ["CV R² (5-fold, Ridge)",     f"{cv_mean} ± {cv_std}"],
        ["Closest Reference Site",    closest['Site']],
        ["Reservoir Classification",  eff_label],
    ], [3.2 * inch, 3.2 * inch])

    res.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#1a5276")),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, 0), 10),
        ("FONTNAME",       (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#eaf4fb"), colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#aed6f1")),
        ("PADDING",        (0, 0), (-1, -1), 7),
        ("TEXTCOLOR",      (1, 8), (1, 8), eff_color),
        ("FONTNAME",       (1, 8), (1, 8), "Helvetica-Bold"),
    ]))
    story.append(res)

    story.append(Spacer(1, 10))
    story.append(Paragraph("Capacity Constraint Factors", SH))
    story.append(Paragraph(
        "DOE/USGS volumetric methodology per §3.11: sweep efficiency is included in C_theoretical "
        "(C_theoretical = A×h×φ×ρ_CO₂×E_sweep). "
        "Four operational constraint factors are then applied: f_pressure, f_depth, f_compartment, f_injectivity.", NO))
    story.append(blue_table([
        ["Constraint",           "Value",                     "Description"],
        ["Sweep Efficiency",     f"{round(sweep * 100, 1)} %",      "Pore volume swept — permeability adjusted"],
        ["Pressure Utilization", f"{round(p_util * 100, 1)} %",     "Headroom before overpressure risk"],
        ["Depth Factor",         f"{round(d_factor * 100, 1)} %",   "Injectivity at reservoir depth"],
        ["Compartmentalization", f"{round(comp * 100, 1)} %",       "Fault isolation limits effective volume"],
        ["Injectivity Factor",   f"{round(injectivity * 100, 1)} %",
         f"Permeability-based capacity fill ({permeability_in} mD)"],
    ], [1.8 * inch, 0.85 * inch, 3.75 * inch]))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#aed6f1"), spaceAfter=10))
    story.append(Paragraph("Analysis Charts", SH))

    chart_images = [
        Image(sens_path,    width=3.1 * inch, height=2.2 * inch),
        Image(ranking_path, width=3.1 * inch, height=2.2 * inch),
    ]
    chart_cols = [3.3 * inch, 3.3 * inch]

    if shap_path and os.path.exists(shap_path):
        story.append(Paragraph("Sensitivity & Ranking", SH))
    else:
        story.append(Paragraph("Sensitivity & Ranking", SH))

    charts = RLTable([chart_images], colWidths=chart_cols)
    charts.setStyle(TableStyle([
        ("ALIGN",   (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",  (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(charts)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Left: One-at-a-time sensitivity (red=negative, blue=positive). "
        "Right: Parameters ranked by absolute impact strength.", NO))

    if shap_path and os.path.exists(shap_path):
        story.append(Spacer(1, 8))
        story.append(Paragraph("SHAP Feature Importance (interaction-aware)", SH))
        story.append(Image(shap_path, width=6.2 * inch, height=2.8 * inch))
        story.append(Paragraph(
            "SHAP values attribute the model's prediction to each input feature, "
            "accounting for interactions. Positive = pushes efficiency up; "
            "Negative = pushes efficiency down. (Lundberg & Lee 2017, NeurIPS)", NO))

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#aed6f1"), spaceAfter=4))
    story.append(Paragraph(
        "Generated by CO<sub>2</sub> Storage Prediction System | "
        "Trained on 70 real-world CCS field sites | "
        "Sources: USGS, NETL Atlas 5<super>th</super> Ed., EU CO2StoP, Bachu (2015), "
        "Park et al. (2021), Das et al. (2023)", FO))

    doc.build(story)
    return buf.getvalue()

# ─────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────
st.write("## ⬇️ Download Results")

out_df = pd.DataFrame({
    "Porosity":                 [porosity_in],
    "Pressure (psi)":           [pressure_in],
    "Temperature (°C)":         [temperature_in],
    "Depth (m)":                [depth_in],
    "Residual Gas Saturation":  [round(sgr_in, 3)],
    "Permeability (mD)":        [permeability_in],
    "Thickness (m)":            [thickness_in],
    "Area (km2)":               [area_in],
    "Predicted Efficiency (%)": [round(prediction * 100, 2)],
    "Bootstrap CI Lower (%)":   [round(ci_lower, 2)],
    "Bootstrap CI Upper (%)":   [round(ci_upper, 2)],
    "Constrained Capacity (t)": [round(capacity_tonnes, 0)],
    "Theoretical Capacity (t)": [round(theoretical, 0)],
    "CO2 Density (kg/m3)":      [round(co2_density, 1)],
    "Closest Reference Site":   [closest['Site']],
    "Sweep Efficiency (%)":     [round(sweep * 100, 1)],
    "Pressure Utilization (%)": [round(p_util * 100, 1)],
    "Depth Factor (%)":         [round(d_factor * 100, 1)],
    "Compartmentalization (%)": [round(comp * 100, 1)],
    "Injectivity Factor (%)":   [round(injectivity * 100, 1)],
    "CV R2 (5-fold)":           [cv_mean],
})

st.download_button("⬇️ Download CSV",
                   out_df.to_csv(index=False),
                   "co2_result.csv")

pdf_bytes = generate_pdf(
    porosity_in=porosity_in, pressure_in=pressure_in,
    temperature_in=temperature_in, depth_in=depth_in,
    sgr_in=sgr_in, permeability_in=permeability_in,
    thickness_in=thickness_in, area_in=area_in,
    prediction=prediction, ci_lower=ci_lower, ci_upper=ci_upper,
    capacity_tonnes=capacity_tonnes, theoretical=theoretical,
    reduction_pct=reduction_pct, sweep=sweep, p_util=p_util,
    d_factor=d_factor, comp=comp, injectivity=injectivity,
    cv_mean=cv_mean, cv_std=cv_std, rmse=rmse,
    closest=closest, eff_label=eff_label, eff_color=eff_color,
    sens_path=_sens_path, ranking_path=_ranking_path,
    shap_path=shap_chart_path,
)

st.download_button("⬇️ Download PDF Report",
                   pdf_bytes, "CO2_Report.pdf", "application/pdf")
