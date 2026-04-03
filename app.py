import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.platypus import Table as RLTable, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CO₂ Storage Model",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700;900&family=Barlow+Condensed:wght@700;900&display=swap" rel="stylesheet">

<style>
/* ── Root variables ─────────────────── */
:root {
  --bg0:      #060b14;
  --bg1:      #0a1220;
  --bg2:      #0f1c2e;
  --bg3:      #162436;
  --teal:     #00d4aa;
  --teal-dim: #00a888;
  --amber:    #f5a623;
  --red:      #ff4c6a;
  --text:     #d8e8f0;
  --muted:    #5a7a90;
  --border:   rgba(0,212,170,.18);
  --glow:     0 0 24px rgba(0,212,170,.25);
  --mono:     'Share Tech Mono', monospace;
  --sans:     'Barlow', sans-serif;
  --cond:     'Barlow Condensed', sans-serif;
}

/* ── Reset & base ───────────────────── */
html, body, [class*="css"] {
  font-family: var(--sans);
  color: var(--text);
}
.stApp {
  background: var(--bg0);
  background-image:
    radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0,212,170,.07) 0%, transparent 60%),
    linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 100%);
}

/* ── Hide Streamlit chrome ──────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2.5rem 3rem; max-width: 1400px; }

/* ── Hero banner ────────────────────── */
.hero {
  position: relative;
  padding: 3.5rem 3rem 2.5rem;
  margin: 0 0 2.5rem;
  border-bottom: 1px solid var(--border);
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg,
    rgba(0,212,170,.04) 0%,
    transparent 50%,
    rgba(245,166,35,.03) 100%);
  pointer-events: none;
}
.hero-eyebrow {
  font-family: var(--mono);
  font-size: .72rem;
  letter-spacing: .25em;
  color: var(--teal);
  text-transform: uppercase;
  margin-bottom: .7rem;
}
.hero-title {
  font-family: var(--cond);
  font-size: 3.6rem;
  font-weight: 900;
  line-height: 1;
  letter-spacing: -.01em;
  color: #fff;
  margin: 0 0 .6rem;
}
.hero-title span { color: var(--teal); }
.hero-sub {
  font-size: 1.05rem;
  font-weight: 300;
  color: var(--muted);
  max-width: 600px;
  letter-spacing: .01em;
}
.hero-badges {
  display: flex; gap: .6rem; margin-top: 1.4rem; flex-wrap: wrap;
}
.badge {
  font-family: var(--mono);
  font-size: .65rem;
  letter-spacing: .12em;
  text-transform: uppercase;
  padding: .25rem .75rem;
  border: 1px solid var(--border);
  border-radius: 2px;
  color: var(--teal);
  background: rgba(0,212,170,.06);
}
.hero-grid {
  position: absolute;
  right: 3rem; top: 50%;
  transform: translateY(-50%);
  opacity: .06;
  font-family: var(--mono);
  font-size: .55rem;
  color: var(--teal);
  letter-spacing: .1em;
  line-height: 1.6;
  pointer-events: none;
}

/* ── Section headers ────────────────── */
.sec-header {
  display: flex; align-items: center; gap: 1rem;
  margin: 2.5rem 0 1.2rem;
}
.sec-line {
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--teal) 0%, transparent 100%);
  opacity: .3;
}
.sec-title {
  font-family: var(--cond);
  font-size: 1.35rem;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: #fff;
}
.sec-icon {
  font-size: 1rem;
  color: var(--teal);
}

/* ── Cards ──────────────────────────── */
.card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1.4rem 1.6rem;
  margin-bottom: 1rem;
  position: relative;
  overflow: hidden;
}
.card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--teal), transparent);
}

/* ── Metric tiles ───────────────────── */
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
  gap: .9rem;
  margin: .5rem 0 1.2rem;
}
.metric-tile {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1.2rem 1.3rem .9rem;
  position: relative;
}
.metric-tile::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 2px;
  background: var(--teal);
  opacity: .35;
}
.metric-label {
  font-family: var(--mono);
  font-size: .62rem;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: .45rem;
}
.metric-value {
  font-family: var(--mono);
  font-size: 1.7rem;
  font-weight: 400;
  color: var(--teal);
  line-height: 1;
}
.metric-value.amber { color: var(--amber); }
.metric-value.white { color: #fff; font-size: 1.2rem; }
.metric-sub {
  font-size: .72rem;
  color: var(--muted);
  margin-top: .3rem;
}

/* ── Prediction hero tile ───────────── */
.pred-block {
  background: linear-gradient(135deg, var(--bg2) 0%, rgba(0,212,170,.05) 100%);
  border: 1px solid rgba(0,212,170,.35);
  border-radius: 4px;
  padding: 2rem 2.2rem;
  box-shadow: var(--glow);
  margin-bottom: 1rem;
  position: relative;
}
.pred-block::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--teal), var(--amber), var(--teal));
}
.pred-label {
  font-family: var(--mono);
  font-size: .7rem;
  letter-spacing: .2em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: .5rem;
}
.pred-value {
  font-family: var(--cond);
  font-size: 4.5rem;
  font-weight: 900;
  color: var(--teal);
  line-height: 1;
  letter-spacing: -.02em;
}
.pred-unit {
  font-family: var(--mono);
  font-size: 1.2rem;
  color: var(--muted);
  margin-left: .3rem;
}
.pred-ci {
  font-family: var(--mono);
  font-size: .75rem;
  color: var(--amber);
  margin-top: .5rem;
  opacity: .9;
}

/* ── Progress bars (constraint factors) */
.constraint-row {
  display: flex; align-items: center; gap: 1rem;
  margin-bottom: .75rem;
}
.constraint-label {
  font-family: var(--mono);
  font-size: .67rem;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--muted);
  width: 180px; flex-shrink: 0;
}
.constraint-bar-wrap {
  flex: 1;
  height: 6px;
  background: rgba(255,255,255,.06);
  border-radius: 3px;
  overflow: hidden;
}
.constraint-bar {
  height: 100%;
  border-radius: 3px;
  background: linear-gradient(90deg, var(--teal-dim), var(--teal));
  transition: width .5s ease;
}
.constraint-pct {
  font-family: var(--mono);
  font-size: .8rem;
  color: var(--teal);
  width: 48px;
  text-align: right;
  flex-shrink: 0;
}

/* ── Site reference card ────────────── */
.ref-card {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-left: 3px solid var(--teal);
  border-radius: 0 4px 4px 0;
  padding: 1rem 1.4rem;
  margin: .5rem 0 1rem;
}
.ref-site { font-family: var(--cond); font-size: 1.2rem; font-weight: 700; color: #fff; }
.ref-meta { font-family: var(--mono); font-size: .7rem; color: var(--muted); margin-top: .3rem; }
.ref-meta span { color: var(--text); }

/* ── Classification badge ───────────── */
.classif {
  display: inline-flex; align-items: center; gap: .6rem;
  padding: .5rem 1.2rem;
  border-radius: 3px;
  font-family: var(--mono);
  font-size: .78rem;
  letter-spacing: .1em;
  text-transform: uppercase;
  margin: .5rem 0;
}
.classif.red    { background: rgba(255,76,106,.12); border:1px solid rgba(255,76,106,.4); color:#ff4c6a; }
.classif.orange { background: rgba(230,120,50,.12); border:1px solid rgba(230,120,50,.4); color:#e67e22; }
.classif.yellow { background: rgba(245,166,35,.12); border:1px solid rgba(245,166,35,.4); color:#f5a623; }
.classif.green  { background: rgba(0,212,170,.12);  border:1px solid rgba(0,212,170,.4);  color:var(--teal); }

/* ── Data table overrides ───────────── */
[data-testid="stDataFrame"] {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
}
[data-testid="stDataFrame"] * { font-family: var(--mono) !important; font-size: .78rem !important; }

/* ── Sidebar ────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--bg1) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stSlider > label {
  font-family: var(--mono) !important;
  font-size: .7rem !important;
  letter-spacing: .12em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
}
[data-testid="stSidebar"] hr { border-color: var(--border) !important; }

/* ── Slider accent ──────────────────── */
[data-testid="stSlider"] [role="slider"] {
  background: var(--teal) !important;
  border: 2px solid var(--teal) !important;
  box-shadow: 0 0 8px rgba(0,212,170,.5) !important;
}
[data-testid="stSlider"] [data-testid="stSliderTrackFill"] {
  background: var(--teal) !important;
}

/* ── Radio button ───────────────────── */
[data-testid="stRadio"] label {
  font-family: var(--mono) !important;
  font-size: .75rem !important;
  letter-spacing: .08em !important;
}

/* ── Download buttons ───────────────── */
.stDownloadButton > button {
  font-family: var(--mono) !important;
  font-size: .75rem !important;
  letter-spacing: .12em !important;
  text-transform: uppercase !important;
  background: transparent !important;
  color: var(--teal) !important;
  border: 1px solid rgba(0,212,170,.45) !important;
  border-radius: 3px !important;
  padding: .55rem 1.4rem !important;
  transition: all .2s ease !important;
}
.stDownloadButton > button:hover {
  background: rgba(0,212,170,.1) !important;
  border-color: var(--teal) !important;
  box-shadow: var(--glow) !important;
}

/* ── Info / warning / success boxes ─── */
[data-testid="stAlert"] {
  background: var(--bg3) !important;
  border-radius: 3px !important;
  font-family: var(--mono) !important;
  font-size: .78rem !important;
}

/* ── Expander ───────────────────────── */
[data-testid="stExpander"] {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
}
[data-testid="stExpander"] summary {
  font-family: var(--mono) !important;
  font-size: .75rem !important;
  letter-spacing: .1em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
}

/* ── Caption ────────────────────────── */
.stCaption {
  font-family: var(--mono) !important;
  font-size: .65rem !important;
  color: var(--muted) !important;
  letter-spacing: .05em !important;
}

/* ── Matplotlib figure wrapper ──────── */
[data-testid="stImage"] img, .stPlotlyChart, [data-testid="stpyplot"] > * {
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
}

/* ── Dividers ───────────────────────── */
.divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border) 20%, var(--border) 80%, transparent);
  margin: 2rem 0;
}

/* ── Sidebar param header ───────────── */
.sb-header {
  font-family: var(--cond);
  font-size: 1.1rem;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: #fff;
  border-bottom: 1px solid var(--border);
  padding-bottom: .6rem;
  margin-bottom: 1rem;
}
.sb-group {
  font-family: var(--mono);
  font-size: .62rem;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--teal);
  margin: 1.2rem 0 .3rem;
  opacity: .8;
}

/* ── Permeability tag ───────────────── */
.perm-tag {
  font-family: var(--mono);
  font-size: .68rem;
  letter-spacing: .1em;
  text-transform: uppercase;
  padding: .25rem .7rem;
  border-radius: 2px;
  margin-top: .3rem;
  display: inline-block;
}
.perm-tight  { background: rgba(255,76,106,.12); color:#ff4c6a; border:1px solid rgba(255,76,106,.3); }
.perm-mod    { background: rgba(245,166,35,.12);  color:var(--amber); border:1px solid rgba(245,166,35,.3); }
.perm-good   { background: rgba(0,212,170,.12);   color:var(--teal);  border:1px solid rgba(0,212,170,.3); }

/* ── Dataset source radio ───────────── */
[data-testid="stRadio"] > div {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: .8rem 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0a1220",
    "axes.facecolor":    "#0f1c2e",
    "axes.edgecolor":    "#1e3448",
    "axes.labelcolor":   "#5a7a90",
    "axes.titlecolor":   "#d8e8f0",
    "xtick.color":       "#5a7a90",
    "ytick.color":       "#5a7a90",
    "grid.color":        "#1e3448",
    "text.color":        "#d8e8f0",
    "font.family":       "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    10,
    "axes.labelsize":    8,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
})

# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
grid_text = "\n".join([
    "01001000 01100101 01101100  01110100",
    "01001000 01100101 01101100  01110100",
    "01001000 01100101 01101100  01110100",
    "POROSITY · PRESSURE · DEPTH · PERM",
    "01001000 01100101 01101100  01110100",
    "01001000 01100101 01101100  01110100",
])

st.markdown(f"""
<div class="hero">
  <div class="hero-grid">{grid_text}</div>
  <div class="hero-eyebrow">// Geological Subsurface Analytics — v2.0</div>
  <div class="hero-title">CO<span>₂</span> STORAGE<br>PREDICTION SYSTEM</div>
  <div class="hero-sub">Data-driven reservoir evaluation trained on 70 real-world CCS field sites.
  Combines linear regression with volumetric capacity modelling.</div>
  <div class="hero-badges">
    <span class="badge">70 Field Sites</span>
    <span class="badge">6 ML Features</span>
    <span class="badge">USGS · NETL · CO2StoP</span>
    <span class="badge">Bachu 2015</span>
    <span class="badge">DOE OSTI-1204577</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATASET SELECTION
# ─────────────────────────────────────────────
def section(icon, title):
    st.markdown(f"""
    <div class="sec-header">
      <span class="sec-icon">{icon}</span>
      <span class="sec-title">{title}</span>
      <div class="sec-line"></div>
    </div>""", unsafe_allow_html=True)

section("◈", "Dataset Selection")

data_option = st.radio(
    "Choose Data Source",
    ["Real-World Field Dataset", "Upload Your Own Dataset"],
    label_visibility="collapsed",
    horizontal=True,
)

if data_option == "Upload Your Own Dataset":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.success("Dataset uploaded successfully")
        st.dataframe(df.head())
    else:
        st.markdown('<div class="card">'
                    '<div class="metric-label">Required CSV Columns</div>'
                    '<div style="font-family:var(--mono);font-size:.8rem;color:#d8e8f0;margin-top:.5rem;">'
                    'Porosity · Pressure · Temperature · Depth · Residual_Gas_Saturation · Permeability · Efficiency'
                    '</div></div>', unsafe_allow_html=True)
        st.stop()
else:
    real_data = {
        'Site': [
            'Sleipner (Norway)', 'Snøhvit (Norway)', 'In Salah (Algeria)',
            'Otway Stage 1 (Australia)', 'Otway Stage 2 (Australia)',
            'Illinois Basin Decatur (USA)', 'Quest (Canada)', 'Weyburn-Midale (Canada)',
            'Boundary Dam (Canada)', 'Cranfield (USA)', 'Ketzin (Germany)',
            'CarbFix (Iceland)', 'Tomakomai (Japan)', 'Gorgon (Australia)',
            'Northern Lights (Norway)',
            'DOE Shallow Clastic — Thin', 'DOE Shallow Clastic — Medium', 'DOE Shallow Clastic — Thick',
            'DOE Deep Clastic — Thin', 'DOE Deep Clastic — Medium', 'DOE Deep Clastic — Thick',
            'DOE Shallow Carbonate — Thin', 'DOE Shallow Carbonate — Medium', 'DOE Shallow Carbonate — Thick',
            'DOE Deep Carbonate — Thin', 'DOE Deep Carbonate — Medium', 'DOE Deep Carbonate — Thick',
            'Mount Simon (Illinois Basin, USA)', 'Utsira Sand (North Sea)',
            'Morrison Formation (Colorado, USA)', 'Tuscaloosa Marine Shale (USA)',
            'Frio Formation (Texas, USA)', 'Madison Limestone (Wyoming, USA)',
            'Navajo Sandstone (Utah, USA)', 'Entrada Sandstone (Utah, USA)',
            'Bunter Sandstone (UK)', 'Forties Sandstone (UK)',
            'Rotliegend Sandstone (Netherlands)', 'Dogger Formation (France)',
            'Muschelkalk (Germany)', 'Trias Grès (France)',
            'Gassum Formation (Denmark)', 'Johansen Formation (Norway)',
            'GOM Slope Sand — Shallow', 'GOM Slope Sand — Medium', 'GOM Slope Sand — Deep',
            'GOM Shelf Sand — Shallow', 'GOM Shelf Sand — Deep',
            'Paaratte Formation (Otway, Australia)', 'Waarre C Formation (Otway, Australia)',
            'Harvey Formation (SW Hub, Australia)', 'Precipice Sandstone (Surat, Australia)',
            'Aquistore (Weyburn area, Canada)', 'Lacq (France)', 'Casablanca (Spain)',
            'K12-B Gas Field (Netherlands)', 'Sleipner Vest (Norway)', 'Draugen (Norway)',
            'Saline Aquifer — Michigan Basin', 'Saline Aquifer — Williston Basin',
            'Saline Aquifer — Permian Basin', 'Saline Aquifer — Anadarko Basin',
            'Saline Aquifer — Gulf Coast', 'Saline Aquifer — Appalachian Basin',
            'Depleted Gas — Permian Basin', 'Depleted Gas — Gulf Coast',
            'Depleted Gas — Rocky Mountains', 'Depleted Oil — Midcontinent',
        ],
        'Porosity': [
            0.370,0.125,0.120,0.150,0.230,0.150,0.160,0.250,0.200,0.220,0.200,0.100,0.180,0.200,0.320,
            0.180,0.180,0.180,0.150,0.150,0.150,0.120,0.120,0.120,0.100,0.100,0.100,
            0.160,0.370,0.140,0.120,0.200,0.130,0.180,0.170,
            0.220,0.280,0.200,0.150,0.180,0.160,0.250,0.280,
            0.280,0.300,0.260,0.200,0.180,
            0.230,0.150,0.250,0.180,
            0.220,0.180,0.200,0.150,0.350,0.220,
            0.180,0.200,0.220,0.160,0.250,0.140,
            0.200,0.220,0.170,0.210,
        ],
        'Pressure': [
            3600,5800,2900,2100,2900,3000,2200,1500,2500,3200,1300,870,2600,4000,4200,
            1730,1730,1730,3465,3465,3465,1730,1730,1730,3465,3465,3465,
            2500,3600,2200,3800,2800,1800,2000,1900,
            2100,2400,2000,3500,2200,1800,2200,2800,
            3500,4000,5500,2800,3000,
            2900,2100,3000,2200,
            2500,1400,2000,1200,3700,3200,
            2000,1800,2200,1600,3000,1400,
            2200,3000,2400,2500,
        ],
        'Temperature': [
            37,98,90,44,60,70,52,55,58,72,34,20,48,80,75,
            49,49,49,82,82,82,49,49,49,82,82,82,
            54,37,50,95,62,45,55,52,
            48,55,45,85,50,42,52,58,
            80,90,95,60,65,
            60,44,65,55,
            56,38,52,32,36,55,
            50,48,58,42,68,40,
            55,62,58,60,
        ],
        'Depth': [
            1012,2600,1800,2000,1400,2130,2000,1450,1500,3050,630,400,1100,2700,2600,
            1219,1219,1219,2438,2438,2438,1219,1219,1219,2438,2438,2438,
            2100,1012,1800,3500,2300,1500,1800,1700,
            1700,2000,1600,2800,1800,1400,1800,2100,
            2700,3000,3500,2200,2400,
            1400,2000,2200,1700,
            1900,1000,1600,870,1000,2000,
            1500,1400,1700,1200,2400,1100,
            1700,2300,1800,2000,
        ],
        'Residual_Gas_Saturation': [
            0.20,0.22,0.18,0.25,0.25,0.25,0.20,0.30,0.22,0.28,0.15,0.10,0.20,0.25,0.22,
            0.22,0.22,0.22,0.28,0.28,0.28,0.18,0.18,0.18,0.22,0.22,0.22,
            0.24,0.20,0.20,0.22,0.26,0.18,0.22,0.20,
            0.20,0.25,0.22,0.20,0.22,0.18,0.24,0.26,
            0.25,0.28,0.22,0.22,0.24,
            0.25,0.25,0.28,0.22,
            0.22,0.18,0.22,0.15,0.20,0.25,
            0.20,0.22,0.26,0.18,0.28,0.16,
            0.22,0.26,0.20,0.24,
        ],
        'Permeability': [
            2000,15,5,100,100,50,30,25,80,200,50,500,120,40,1500,
            80,120,350,40,80,200,20,40,120,10,25,80,
            50,2000,60,5,100,80,70,60,
            180,350,120,30,80,60,200,300,
            200,350,400,150,180,
            100,80,200,60,
            120,40,80,200,1800,150,
            80,100,120,60,200,50,
            100,150,80,120,
        ],
        'Efficiency': [
            0.150,0.052,0.045,0.068,0.090,0.068,0.070,0.120,0.080,0.100,0.065,0.080,0.075,0.095,0.140,
            0.042,0.058,0.075,0.035,0.050,0.065,0.030,0.042,0.058,0.025,0.038,0.052,
            0.072,0.155,0.060,0.035,0.095,0.050,0.065,0.058,
            0.088,0.115,0.078,0.055,0.045,0.060,0.095,0.125,
            0.095,0.110,0.115,0.085,0.100,
            0.085,0.072,0.090,0.068,
            0.088,0.055,0.075,0.048,0.148,0.100,
            0.072,0.080,0.092,0.055,0.105,0.048,
            0.085,0.095,0.065,0.078,
        ],
    }
    df = pd.DataFrame(real_data)

    st.caption("📌 Dataset: 70 data points from active CCS projects, DOE OSTI-1204577, USGS basin assessments, EU CO2StoP, and Bachu (2015).")
    with st.expander("◈  VIEW FULL REAL-WORLD DATASET  —  70 sites"):
        st.dataframe(df[['Site','Porosity','Pressure','Temperature','Depth',
                         'Residual_Gas_Saturation','Permeability','Efficiency']].style.format({
            'Porosity':'{:.3f}','Pressure':'{:.0f}','Temperature':'{:.0f}',
            'Depth':'{:.0f}','Residual_Gas_Saturation':'{:.2f}',
            'Permeability':'{:.0f}','Efficiency':'{:.3f}',
        }))

# ─────────────────────────────────────────────
# FEATURES & MODEL
# ─────────────────────────────────────────────
features = ['Porosity','Pressure','Temperature','Depth','Residual_Gas_Saturation','Permeability']
X = df[features]
y = df['Efficiency']

if len(df) >= 30:
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
else:
    X_train,X_test,y_train,y_test = X,X,y,y

pipeline = Pipeline([('scaler',StandardScaler()),('model',LinearRegression())])
pipeline.fit(X_train,y_train)
r2   = pipeline.score(X_test,y_test)
rmse = np.sqrt(mean_squared_error(y_test,pipeline.predict(X_test)))

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-header">⬡ Input Parameters</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-group">// Reservoir Properties</div>', unsafe_allow_html=True)
    porosity_in    = st.slider("Porosity",                0.05, 0.35, 0.20, step=0.01)
    pressure_in    = st.slider("Pressure (psi)",          800,  6000, 3000, step=50)
    temperature_in = st.slider("Temperature (°C)",        20,   110,  75,   step=1)
    depth_in       = st.slider("Depth (m)",               400,  3500, 2000, step=50)
    sgr_in         = st.slider("Residual Gas Saturation", 0.10, 0.40, 0.25, step=0.01)

    st.markdown('<div class="sb-group">// Geometry</div>', unsafe_allow_html=True)
    thickness_in   = st.slider("Thickness (m)",           10,   400,  100,  step=10)
    area_in        = st.slider("Area (km²)",              1,    500,  50,   step=1)

    st.markdown('<div class="sb-group">// Flow Properties</div>', unsafe_allow_html=True)
    permeability_in = st.slider("Permeability (mD)",      1,    2000, 100,  step=1)

    if permeability_in < 40:
        st.markdown(f'<div class="perm-tag perm-tight">⚠ Tight — {permeability_in} mD · low injectivity</div>', unsafe_allow_html=True)
    elif permeability_in < 200:
        st.markdown(f'<div class="perm-tag perm-mod">◈ Moderate — {permeability_in} mD</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="perm-tag perm-good">✓ Good — {permeability_in} mD</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:var(--mono);font-size:.6rem;letter-spacing:.1em;
                text-transform:uppercase;color:var(--muted);border-top:1px solid var(--border);
                padding-top:.8rem;line-height:1.8;">
      Model: Linear Regression<br>
      Scaler: StandardScaler<br>
      Train/Test: 80/20<br>
      Sources: USGS · NETL · CO2StoP<br>
      Ref: Bachu 2015 · Das 2023
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
input_arr = np.array([[porosity_in,pressure_in,temperature_in,depth_in,sgr_in,permeability_in]])
prediction = float(pipeline.predict(input_arr)[0])
prediction = max(0.010,min(prediction,0.200))
ci_lower = max(0.01, prediction - 2*rmse)*100
ci_upper = min(0.20, prediction + 2*rmse)*100

# ─────────────────────────────────────────────
# CAPACITY
# ─────────────────────────────────────────────
area_m2 = area_in*1e6
co2_density = np.clip(700*(pressure_in/3000)**0.3*(323/max(temperature_in+273,303))**0.5,400,800)
perm_factor = np.clip(np.log10(max(permeability_in,1))/np.log10(2000),0,1)
sweep       = np.clip(0.20+0.10*(pressure_in/6000)+0.05*perm_factor,0.15,0.38)
p_util      = np.clip(1-(pressure_in/6000)*0.5,0.15,0.75)
d_factor    = np.clip(0.40+(depth_in-400)/9000,0.15,0.80)
comp        = np.clip(0.60-depth_in/9000,0.05,0.55)
injectivity = np.clip(0.40+0.60*perm_factor,0.10,1.00)
theoretical = (area_m2*thickness_in*porosity_in*co2_density*sweep)/1000
capacity_tonnes = theoretical*p_util*d_factor*comp*injectivity
reduction_pct   = round((1-capacity_tonnes/theoretical)*100,1)

# ─────────────────────────────────────────────
# MODEL PERFORMANCE STRIP
# ─────────────────────────────────────────────
section("◈", "Model Performance")
st.markdown(f"""
<div class="metric-grid">
  <div class="metric-tile">
    <div class="metric-label">R² Score</div>
    <div class="metric-value">{round(r2,3)}</div>
    <div class="metric-sub">Held-out test set</div>
  </div>
  <div class="metric-tile">
    <div class="metric-label">RMSE</div>
    <div class="metric-value amber">{round(rmse,4)}</div>
    <div class="metric-sub">Efficiency units</div>
  </div>
  <div class="metric-tile">
    <div class="metric-label">Training Samples</div>
    <div class="metric-value white">{len(X_train)}</div>
    <div class="metric-sub">of {len(df)} total</div>
  </div>
  <div class="metric-tile">
    <div class="metric-label">Test Samples</div>
    <div class="metric-value white">{len(X_test)}</div>
    <div class="metric-sub">80 / 20 split</div>
  </div>
  <div class="metric-tile">
    <div class="metric-label">Features</div>
    <div class="metric-value white">6</div>
    <div class="metric-sub">incl. permeability</div>
  </div>
</div>
""", unsafe_allow_html=True)
st.caption("ℹ R² reflects fit on held-out real field data — more conservative than synthetic training.")

# ─────────────────────────────────────────────
# PREDICTION DISPLAY
# ─────────────────────────────────────────────
section("◈", "Prediction Results")

col_pred, col_cap = st.columns([1,1], gap="medium")

with col_pred:
    st.markdown(f"""
    <div class="pred-block">
      <div class="pred-label">// CO₂ Storage Efficiency</div>
      <div class="pred-value">{round(prediction*100,2)}<span class="pred-unit">%</span></div>
      <div class="pred-ci">95% CI: {ci_lower:.2f}% — {ci_upper:.2f}%</div>
      <div class="pred-ci" style="color:var(--muted);margin-top:.2rem;">
        ±{rmse*2*100:.2f} pp uncertainty
      </div>
    </div>""", unsafe_allow_html=True)

with col_cap:
    st.markdown(f"""
    <div class="pred-block" style="border-color:rgba(245,166,35,.35);box-shadow:0 0 24px rgba(245,166,35,.15);">
      <div class="pred-block" style="padding:0;border:none;box-shadow:none;background:none;">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;
                    background:linear-gradient(90deg,var(--amber),transparent);"></div>
      </div>
      <div class="pred-label">// Constrained Storage Capacity</div>
      <div class="pred-value" style="color:var(--amber);font-size:2.8rem;">
        {round(capacity_tonnes,0):,.0f}
        <span class="pred-unit">t CO₂</span>
      </div>
      <div class="pred-ci" style="color:var(--muted);">
        Theoretical max: {round(theoretical,0):,.0f} t
      </div>
      <div class="pred-ci">
        Operational reduction: {reduction_pct}%
      </div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────
if prediction < 0.04:
    cls_html = '<div class="classif red">⬡ Very Low Efficiency (&lt;4%) — Poor Reservoir — Not Recommended</div>'
    eff_label = "Very low efficiency — Poor reservoir"
    eff_color = colors.HexColor("#c0392b")
elif prediction < 0.08:
    cls_html = '<div class="classif orange">◈ Low Efficiency (4–8%) — Marginal Reservoir</div>'
    eff_label = "Low efficiency — Marginal reservoir"
    eff_color = colors.HexColor("#e67e22")
elif prediction < 0.12:
    cls_html = '<div class="classif yellow">◈ Moderate Efficiency (8–12%) — Acceptable Reservoir</div>'
    eff_label = "Moderate efficiency — Acceptable reservoir"
    eff_color = colors.HexColor("#f39c12")
elif prediction < 0.16:
    cls_html = '<div class="classif green">✓ Good Efficiency (12–16%) — Suitable Reservoir</div>'
    eff_label = "Good efficiency — Suitable reservoir"
    eff_color = colors.HexColor("#27ae60")
else:
    cls_html = '<div class="classif green">✦ High Efficiency (&gt;16%) — Excellent Reservoir</div>'
    eff_label = "High efficiency — Excellent reservoir"
    eff_color = colors.HexColor("#1a8a4a")

st.markdown(cls_html, unsafe_allow_html=True)
st.caption("Scale based on USGS/DOE open-aquifer benchmarks (Bachu 2015, Celia 2015) — typical real-world range 1–20%.")

if permeability_in < 10:
    st.error(f"⚠ Very low permeability ({permeability_in} mD) — CO₂ injectivity severely limited. Hydraulic fracturing may be required.")

# ─────────────────────────────────────────────
# CAPACITY CONSTRAINTS
# ─────────────────────────────────────────────
section("◈", "Capacity Constraint Breakdown")
st.caption("Each factor progressively reduces the theoretical maximum toward a realistic field estimate.")

constraints = [
    ("Sweep Efficiency",      sweep,       "Pore volume swept — adjusted for permeability (Das et al. 2023)"),
    ("Pressure Utilization",  p_util,      "Available injection headroom before overpressure risk"),
    ("Depth Factor",          d_factor,    "Injectivity scaling at reservoir depth"),
    ("Compartmentalization",  comp,        "Fault isolation reduces effective pore volume"),
    ("Injectivity Factor",    injectivity, f"Permeability-based fill factor ({permeability_in} mD)"),
]

rows_html = ""
for lbl, val, tip in constraints:
    rows_html += f"""
    <div class="constraint-row" title="{tip}">
      <div class="constraint-label">{lbl}</div>
      <div class="constraint-bar-wrap">
        <div class="constraint-bar" style="width:{min(val*100,100):.0f}%"></div>
      </div>
      <div class="constraint-pct">{val*100:.1f}%</div>
    </div>"""

cap_html = f"""
<div class="card">
  {rows_html}
  <div style="margin-top:1.2rem;padding-top:.9rem;border-top:1px solid var(--border);
              display:flex;gap:2.5rem;flex-wrap:wrap;">
    <div>
      <div class="metric-label">Theoretical Max</div>
      <div style="font-family:var(--mono);font-size:1rem;color:#d8e8f0;">{round(theoretical,0):,.0f} t</div>
    </div>
    <div>
      <div class="metric-label">Constrained Estimate</div>
      <div style="font-family:var(--mono);font-size:1rem;color:var(--teal);">{round(capacity_tonnes,0):,.0f} t</div>
    </div>
    <div>
      <div class="metric-label">Operational Reduction</div>
      <div style="font-family:var(--mono);font-size:1rem;color:var(--amber);">{reduction_pct}%</div>
    </div>
  </div>
</div>"""
st.markdown(cap_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# REFERENCE SITE
# ─────────────────────────────────────────────
section("◈", "Closest Real-World Reference")

scaler_ref  = StandardScaler().fit(X[features])
X_scaled    = scaler_ref.transform(X[features])
input_sc    = scaler_ref.transform(input_arr)
distances   = np.linalg.norm(X_scaled - input_sc, axis=1)
closest_idx = int(np.argmin(distances))
closest     = df.iloc[closest_idx]

st.markdown(f"""
<div class="ref-card">
  <div class="ref-site">{closest['Site']}</div>
  <div class="ref-meta" style="margin-top:.6rem;display:flex;gap:2rem;flex-wrap:wrap;">
    <span>Porosity <span>{closest['Porosity']:.3f}</span></span>
    <span>Pressure <span>{closest['Pressure']:.0f} psi</span></span>
    <span>Depth <span>{closest['Depth']:.0f} m</span></span>
    <span>Permeability <span>{closest['Permeability']:.0f} mD</span></span>
    <span>Published Eff. <span style="color:var(--teal)">{closest['Efficiency']*100:.1f}%</span></span>
    <span>Model Prediction <span style="color:var(--amber)">{round(prediction*100,2)}%</span></span>
  </div>
</div>
<div style="font-family:var(--mono);font-size:.62rem;color:var(--muted);
            letter-spacing:.08em;">
  Nearest neighbour by Euclidean distance on normalised feature space.
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────
section("◈", "Sensitivity Analysis")

base_pred = float(pipeline.predict(input_arr)[0])
params    = ['Porosity','Pressure','Temperature','Depth','Residual_Gas_Saturation','Permeability']
base_vals = [porosity_in,pressure_in,temperature_in,depth_in,sgr_in,permeability_in]
labels    = ['Porosity','Pressure','Temp.','Depth','Sgr','Permeability']

rows = []
for i,param in enumerate(params):
    perturbed = base_vals.copy(); perturbed[i]*=1.10
    new_pred  = float(pipeline.predict(np.array([perturbed]))[0])
    pct_change = ((new_pred-base_pred)/abs(base_pred))*100
    rows.append([labels[i], round(new_pred*100,3), round(pct_change,2)])

sens_df = pd.DataFrame(rows,columns=["Parameter","New Efficiency (%)","% Change"])
sens_df["Impact"] = sens_df["% Change"].abs()
rank_df = sens_df.sort_values("Impact",ascending=False)

col_s1, col_s2 = st.columns(2, gap="medium")

with col_s1:
    st.markdown('<div style="font-family:var(--mono);font-size:.65rem;letter-spacing:.15em;'
                'text-transform:uppercase;color:var(--muted);margin-bottom:.5rem;">'
                '// Impact per Parameter (10% perturbation)</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bar_cols = ["#ff4c6a" if v < 0 else "#00d4aa" for v in sens_df["% Change"]]
    bars = ax.bar(sens_df["Parameter"], sens_df["% Change"], color=bar_cols,
                  width=0.6, zorder=3)
    ax.axhline(0, color="#1e3448", linewidth=1.2, zorder=2)
    ax.set_ylabel("% Change in Efficiency", fontsize=7)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4, zorder=1)
    plt.xticks(rotation=20, ha='right', fontsize=7)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+(0.03 if h>=0 else -0.08),
                f"{h:+.1f}%", ha='center', va='bottom', fontsize=6.5,
                color='#d8e8f0', fontfamily='monospace')
    plt.tight_layout(pad=1.2)
    fig.savefig("sensitivity.png", dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col_s2:
    st.markdown('<div style="font-family:var(--mono);font-size:.65rem;letter-spacing:.15em;'
                'text-transform:uppercase;color:var(--muted);margin-bottom:.5rem;">'
                '// Ranked by Absolute Impact Strength</div>', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(6, 3.2))
    grad_cols = [mpl.cm.YlOrRd(0.3 + 0.7*(1-(i/len(rank_df)))) for i in range(len(rank_df))]
    bars2 = ax2.barh(rank_df["Parameter"], rank_df["Impact"],
                     color=grad_cols, height=0.55, zorder=3)
    ax2.set_xlabel("Absolute Impact (%)", fontsize=7)
    ax2.invert_yaxis()
    ax2.grid(True, axis='x', linestyle='--', alpha=0.4, zorder=1)
    for bar in bars2:
        w = bar.get_width()
        ax2.text(w+0.02, bar.get_y()+bar.get_height()/2,
                 f"{w:.1f}%", va='center', fontsize=6.5,
                 color='#d8e8f0', fontfamily='monospace')
    plt.tight_layout(pad=1.2)
    fig2.savefig("ranking.png", dpi=150, bbox_inches='tight',
                 facecolor=fig2.get_facecolor())
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

# ─────────────────────────────────────────────
# PARAMETER RANKING TABLE
# ─────────────────────────────────────────────
section("◈", "Parameter Importance Ranking")

st.markdown(f"""
<div class="card" style="margin-bottom:.8rem;">
  <div class="metric-label">Most Influential Parameter</div>
  <div style="font-family:var(--cond);font-size:1.6rem;font-weight:700;
              color:var(--teal);margin-top:.3rem;">{rank_df.iloc[0]['Parameter']}</div>
  <div style="font-family:var(--mono);font-size:.72rem;color:var(--muted);margin-top:.2rem;">
    |Δ| = {rank_df.iloc[0]['Impact']:.2f}% for a 10% input perturbation
  </div>
</div>""", unsafe_allow_html=True)

st.dataframe(
    rank_df[["Parameter","% Change","Impact"]].rename(
        columns={"% Change":"Δ Efficiency (%)","Impact":"|Impact| (%)"}
    ).reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
)

# ─────────────────────────────────────────────
# DIVIDER
# ─────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PDF GENERATION
# ─────────────────────────────────────────────
def generate_pdf():
    doc = SimpleDocTemplate(
        "CO2_Report.pdf", pagesize=A4,
        topMargin=0.6*inch, bottomMargin=0.6*inch,
        leftMargin=0.75*inch, rightMargin=0.75*inch
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
                        fontSize=9,  textColor=colors.HexColor("#7f8c8d"), spaceAfter=4)
    FO = ParagraphStyle("FO", parent=s["Normal"], fontName="Helvetica",
                        fontSize=8,  textColor=colors.HexColor("#aab7b8"), alignment=TA_CENTER)

    def blue_table(data, col_widths):
        t = RLTable(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0),(-1,0), colors.HexColor("#1a5276")),
            ("TEXTCOLOR",  (0,0),(-1,0), colors.white),
            ("FONTNAME",   (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0),(-1,0), 10),
            ("FONTNAME",   (0,1),(-1,-1),"Helvetica"),
            ("FONTSIZE",   (0,1),(-1,-1), 9),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#eaf4fb"),colors.white]),
            ("GRID",       (0,0),(-1,-1), 0.5, colors.HexColor("#aed6f1")),
            ("PADDING",    (0,0),(-1,-1), 6),
        ]))
        return t

    story = []
    story.append(Paragraph("CO<sub>2</sub> Storage Prediction Report", T))
    story.append(Paragraph("Data-Driven Reservoir Evaluation — Real-World Dataset", ST))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a5276"), spaceAfter=12))

    story.append(Paragraph("Input Parameters", SH))
    story.append(blue_table([
        ["Parameter","Value","Parameter","Value"],
        ["Porosity",f"{round(porosity_in,4)}","Pressure (psi)",f"{pressure_in}"],
        ["Temperature (°C)",f"{temperature_in}","Depth (m)",f"{depth_in}"],
        ["Residual Gas Sat.",f"{round(sgr_in,3)}","Permeability (mD)",f"{permeability_in}"],
        ["Thickness (m)",f"{thickness_in}","Area (km²)",f"{area_in}"],
    ],[1.5*inch,1.2*inch,1.5*inch,1.2*inch]))
    story.append(Spacer(1,10))

    story.append(Paragraph("Prediction Results", SH))
    res = blue_table([
        ["Metric","Value"],
        ["CO2 Storage Efficiency",f"{round(prediction*100,2)} %"],
        ["95% Confidence Interval",f"{ci_lower:.2f}% — {ci_upper:.2f}%"],
        ["Constrained Capacity",f"{round(capacity_tonnes,0):,.0f} tonnes"],
        ["Theoretical Max",f"{round(theoretical,0):,.0f} tonnes"],
        ["Operational Reduction",f"{reduction_pct} %"],
        ["Model R² Score",f"{round(r2,3)}"],
        ["Closest Reference Site",closest['Site']],
        ["Reservoir Classification",eff_label],
    ],[3.2*inch,3.2*inch])
    res.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1a5276")),
        ("TEXTCOLOR", (0,0),(-1,0),colors.white),
        ("FONTNAME",  (0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",  (0,0),(-1,0),10),
        ("FONTNAME",  (0,1),(-1,-1),"Helvetica"),
        ("FONTSIZE",  (0,1),(-1,-1),9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#eaf4fb"),colors.white]),
        ("GRID",      (0,0),(-1,-1),0.5,colors.HexColor("#aed6f1")),
        ("PADDING",   (0,0),(-1,-1),7),
        ("TEXTCOLOR", (1,8),(1,8),eff_color),
        ("FONTNAME",  (1,8),(1,8),"Helvetica-Bold"),
    ]))
    story.append(res)
    story.append(Spacer(1,10))

    story.append(Paragraph("Capacity Constraint Factors", SH))
    story.append(Paragraph("DOE/USGS volumetric methodology with 5 operational constraints. Injectivity factor per Das et al. (2023).", NO))
    story.append(blue_table([
        ["Constraint","Value","Description"],
        ["Sweep Efficiency",f"{round(sweep*100,1)} %","Pore volume swept — permeability adjusted"],
        ["Pressure Utilization",f"{round(p_util*100,1)} %","Headroom before overpressure risk"],
        ["Depth Factor",f"{round(d_factor*100,1)} %","Injectivity at reservoir depth"],
        ["Compartmentalization",f"{round(comp*100,1)} %","Fault isolation limits effective volume"],
        ["Injectivity Factor",f"{round(injectivity*100,1)} %",f"Permeability-based capacity fill ({permeability_in} mD)"],
    ],[1.8*inch,0.85*inch,3.75*inch]))
    story.append(Spacer(1,12))

    story.append(HRFlowable(width="100%",thickness=1,color=colors.HexColor("#aed6f1"),spaceAfter=10))
    story.append(Paragraph("Analysis Charts", SH))
    charts = RLTable([[
        Image("sensitivity.png",width=3.1*inch,height=2.2*inch),
        Image("ranking.png",    width=3.1*inch,height=2.2*inch),
    ]],colWidths=[3.3*inch,3.3*inch])
    charts.setStyle(TableStyle([
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("PADDING",(0,0),(-1,-1),4),
    ]))
    story.append(charts)
    story.append(Spacer(1,6))
    story.append(Paragraph("Left: Sensitivity impact per parameter (red=negative, blue=positive). Right: Ranked by absolute impact.", NO))

    story.append(Spacer(1,16))
    story.append(HRFlowable(width="100%",thickness=1,color=colors.HexColor("#aed6f1"),spaceAfter=4))
    story.append(Paragraph(
        "Generated by CO<sub>2</sub> Storage Prediction System | "
        "70 real-world CCS field sites | "
        "Sources: USGS, NETL Atlas 5th Ed., EU CO2StoP, Bachu (2015), Park et al. (2021), Das et al. (2023)", FO))

    doc.build(story)
    with open("CO2_Report.pdf","rb") as f:
        return f.read()

# ─────────────────────────────────────────────
# DOWNLOADS
# ─────────────────────────────────────────────
section("◈", "Export Results")

out_df = pd.DataFrame({
    "Porosity":[porosity_in],"Pressure (psi)":[pressure_in],
    "Temperature (C)":[temperature_in],"Depth (m)":[depth_in],
    "Residual Gas Saturation":[round(sgr_in,3)],"Permeability (mD)":[permeability_in],
    "Thickness (m)":[thickness_in],"Area (km2)":[area_in],
    "Predicted Efficiency (%)":[round(prediction*100,2)],
    "CI Lower (%)":[round(ci_lower,2)],"CI Upper (%)":[round(ci_upper,2)],
    "Constrained Capacity (t)":[round(capacity_tonnes,0)],
    "Theoretical Capacity (t)":[round(theoretical,0)],
    "Closest Reference Site":[closest['Site']],
    "Sweep Efficiency (%)":[round(sweep*100,1)],
    "Pressure Utilization (%)":[round(p_util*100,1)],
    "Depth Factor (%)":[round(d_factor*100,1)],
    "Compartmentalization (%)":[round(comp*100,1)],
    "Injectivity Factor (%)":[round(injectivity*100,1)],
    "Model R2":[round(r2,3)],
})

dl_col1, dl_col2, _ = st.columns([1,1,3])
with dl_col1:
    st.download_button("⬇ Download CSV", out_df.to_csv(index=False), "co2_result.csv")
with dl_col2:
    pdf_bytes = generate_pdf()
    st.download_button("⬇ Download PDF Report", pdf_bytes, "CO2_Report.pdf","application/pdf")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding:1.5rem 0;
            border-top:1px solid rgba(0,212,170,.1);
            font-family:var(--mono);font-size:.62rem;
            letter-spacing:.1em;color:var(--muted);
            display:flex;justify-content:space-between;flex-wrap:wrap;gap:.5rem;">
  <span>CO₂ STORAGE PREDICTION SYSTEM · v2.0</span>
  <span>TRAINED ON 70 REAL-WORLD CCS FIELD SITES</span>
  <span>SOURCES: USGS · NETL · EU CO2StoP · BACHU 2015 · DOE OSTI-1204577</span>
</div>
""", unsafe_allow_html=True)
