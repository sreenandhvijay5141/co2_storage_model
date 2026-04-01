import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.platypus import Table as RLTable, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="CO2 Storage Model", layout="wide")

st.title("🌍 CO₂ Storage Prediction System")
st.markdown("### Data-Driven Reservoir Evaluation")

# -----------------------------
# DATASET SELECTION
# -----------------------------
st.write("## 📂 Dataset Selection")

data_option = st.radio(
    "Choose Data Source",
    ["Synthetic Dataset (Real-World Calibrated)", "Upload Real Dataset"]
)

if data_option == "Upload Real Dataset":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.success("Dataset uploaded successfully")
        st.dataframe(df.head())
    else:
        st.info("Please upload a CSV with columns: Porosity, Pressure, Temperature, Depth, Residual_Gas_Saturation, Efficiency")
        st.stop()

else:
    # ---------------------------------------------------------------
    # REAL-WORLD CALIBRATED SYNTHETIC DATASET
    # Based on published CCS reservoir data:
    #   - USGS National CO2 Storage Assessment (33 US basins)
    #   - European CO2StoP database (EU member states)
    #   - Sleipner (Norway): porosity 0.35, depth 1012m, efficiency ~0.15
    #   - Snohvit (Norway): porosity 0.125, depth 2600m, efficiency ~0.05
    #   - Illinois Basin Decatur (USA): porosity 0.15, depth 2130m
    #   - Mount Simon Sandstone: porosity range 0.08-0.22 with depth
    #   - Literature range: efficiency 0.01-0.20 for open aquifers (Bachu 2015)
    # ---------------------------------------------------------------
    np.random.seed(42)
    n = 200  # Increased from 120 for better training

    # Realistic parameter ranges from literature
    depth       = np.random.uniform(800, 3500, n)      # Must be >800m for supercritical CO2
    porosity    = np.clip(
                    16.36 * np.exp(-0.00039 * depth) / 100   # Mount Simon depth-porosity relationship
                    + np.random.normal(0, 0.02, n),
                    0.05, 0.35
                  )
    # Pressure follows hydrostatic gradient ~0.43 psi/ft (~9.8 kPa/m)
    pressure    = depth * 1.45 + np.random.normal(0, 150, n)   # psi
    pressure    = np.clip(pressure, 1200, 6000)

    # Temperature follows geothermal gradient ~25-30°C/km from ~15°C surface
    temperature = 15 + (depth / 1000) * 27 + np.random.normal(0, 3, n)
    temperature = np.clip(temperature, 30, 110)

    sgr         = np.random.uniform(0.10, 0.40, n)   # Residual gas saturation

    data = {
        'Porosity':                 porosity,
        'Pressure':                 pressure,
        'Temperature':              temperature,
        'Depth':                    depth,
        'Residual_Gas_Saturation':  sgr,
    }

    df = pd.DataFrame(data)

    # ---------------------------------------------------------------
    # REAL-WORLD CALIBRATED EFFICIENCY FORMULA
    # Based on DOE/USGS volumetric efficiency methodology:
    #   - Open aquifer efficiency: 1-20% (Bachu 2015, Celia 2015)
    #   - Deeper = denser supercritical CO2 = slightly better efficiency
    #   - Higher pressure = better supercritical state
    #   - Higher temperature = lower CO2 density = slightly lower efficiency
    #   - Higher porosity = more pore space but not linearly more efficient
    #   - SGR contributes to residual trapping (positive effect)
    # ---------------------------------------------------------------
    df['Efficiency'] = (
        0.02                                              # Base efficiency (2% floor - open aquifers)
        + 0.08  * (df['Porosity'] / 0.30)                # Porosity contribution (normalized)
        + 0.03  * np.clip((df['Depth'] - 800) / 2700, 0, 1)   # Depth bonus (supercritical density)
        + 0.025 * np.clip((df['Pressure'] - 1200) / 4800, 0, 1)  # Pressure contribution
        - 0.02  * np.clip((df['Temperature'] - 30) / 80, 0, 1)   # Temperature penalty
        + 0.04  * df['Residual_Gas_Saturation']           # Residual trapping contribution
        + np.random.normal(0, 0.008, n)                   # Small realistic noise
    )

    # Clip to realistic range: 1% to 20% (literature-backed)
    df['Efficiency'] = np.clip(df['Efficiency'], 0.01, 0.20)

    st.caption("📌 Training data calibrated to real-world CCS ranges: USGS (33 US basins), EU CO2StoP, Sleipner & Snøhvit projects.")

# -----------------------------
# FEATURES & MODEL
# -----------------------------
X = df[['Porosity', 'Pressure', 'Temperature', 'Depth', 'Residual_Gas_Saturation']]
y = df['Efficiency']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# SIDEBAR INPUTS
# Use real-world literature ranges for sliders
# -----------------------------
st.sidebar.header("🔧 Input Parameters")

porosity_in    = st.sidebar.slider("Porosity",                 0.05,  0.35,  0.20,  step=0.01)
pressure_in    = st.sidebar.slider("Pressure (psi)",           1200,  6000,  3000,  step=50)
temperature_in = st.sidebar.slider("Temperature (°C)",         30,    110,   75,    step=1)
depth_in       = st.sidebar.slider("Depth (m)",                800,   3500,  2000,  step=50)
sgr_in         = st.sidebar.slider("Residual Gas Saturation",  0.10,  0.40,  0.25,  step=0.01)
thickness_in   = st.sidebar.slider("Reservoir Thickness (m)",  10,    400,   100,   step=10)
area_in        = st.sidebar.slider("Reservoir Area (km²)",     1,     500,   50,    step=1)

# -----------------------------
# PERFORMANCE
# -----------------------------
r2   = model.score(X_test, y_test)
rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

st.write("## 📊 Model Performance")
col1, col2 = st.columns(2)
col1.metric("R² Score", round(r2, 3))
col2.metric("RMSE", round(rmse, 4))

# -----------------------------
# PREDICTION
# -----------------------------
input_data = np.array([[porosity_in, pressure_in, temperature_in, depth_in, sgr_in]])
prediction = float(model.predict(input_data)[0])
prediction = max(0.01, min(prediction, 0.20))   # Clip to realistic range

# -----------------------------
# CAPACITY CALCULATION
# Industry-standard DOE volumetric formula:
#   M = A × h × φ × ρ_CO2 × E
# With real-world constraint factors
# -----------------------------
area_m2 = area_in * 1e6

# CO2 density at in-situ conditions (supercritical approximation)
# At depth >800m: density ~500-800 kg/m3 depending on P and T
co2_density = 700 * (pressure_in / 3000) ** 0.3 * (50 / max(temperature_in, 50)) ** 0.2
co2_density = max(400, min(co2_density, 800))

# Sweep efficiency — industry standard 20-40% of pore volume swept
# Based on NETL efficiency factor methodology
sweep_efficiency = 0.20 + 0.15 * (pressure_in / 6000)   # 20-35% range
sweep_efficiency = max(0.15, min(sweep_efficiency, 0.35))

# Pressure utilization — how much headroom before overpressure
# High pressure reservoirs have less injection headroom
max_safe_pressure = 0.8 * pressure_in   # 80% of current pressure as limit (industry rule)
pressure_utilization = 1 - (pressure_in / 6000) * 0.5
pressure_utilization = max(0.15, min(pressure_utilization, 0.75))

# Depth factor — supercritical CO2 more stable at depth, but injectivity drops
depth_factor = min(0.85, 0.4 + (depth_in - 800) / 8000)
depth_factor = max(0.15, depth_factor)

# Compartmentalization factor — fault isolation reduces effective volume
# Calibrated: Snohvit had ~6% utilization; Sleipner had ~0.003% (huge aquifer)
compartment_factor = max(0.05, 0.6 - (depth_in / 8000))

# Theoretical max (no operational constraints — just volumetric)
theoretical_capacity = (area_m2 * thickness_in * porosity_in * co2_density * sweep_efficiency) / 1000

# Fully constrained realistic capacity
capacity_kg = (
    area_m2 * thickness_in * porosity_in * co2_density
    * sweep_efficiency
    * pressure_utilization
    * depth_factor
    * compartment_factor
)
capacity_tonnes = capacity_kg / 1000

reduction_pct = round((1 - capacity_tonnes / theoretical_capacity) * 100, 1)

# -----------------------------
# DISPLAY PREDICTION
# -----------------------------
st.write("## 🎯 Prediction")
col1, col2 = st.columns(2)
col1.metric("CO₂ Storage Efficiency", f"{round(prediction * 100, 2)}%")
col2.metric("CO₂ Storage Capacity (tonnes)", f"{round(capacity_tonnes, 0):,.0f}")

# -----------------------------
# CAPACITY CONSTRAINT BREAKDOWN
# -----------------------------
st.write("## 🔍 Capacity Constraint Breakdown")
st.caption("Shows how each real-world factor reduces the theoretical maximum capacity.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Sweep Efficiency",      f"{round(sweep_efficiency * 100, 1)}%",
            help="% of pore volume actually swept by CO2 (industry standard: 20-35%)")
col2.metric("Pressure Utilization",  f"{round(pressure_utilization * 100, 1)}%",
            help="Headroom before overpressure risk")
col3.metric("Depth Factor",          f"{round(depth_factor * 100, 1)}%",
            help="Injectivity at this depth")
col4.metric("Compartmentalization",  f"{round(compartment_factor * 100, 1)}%",
            help="Fault isolation reduces effective volume")

st.info(
    f"📌 Theoretical max (volumetric only): **{round(theoretical_capacity, 0):,.0f} tonnes**  \n"
    f"✅ Constrained realistic estimate: **{round(capacity_tonnes, 0):,.0f} tonnes**  \n"
    f"📉 Operational reduction: **{reduction_pct}%**"
)

# -----------------------------
# INTERPRETATION
# Based on Bachu 2015 / USGS efficiency classification
# -----------------------------
st.write("## 📘 Interpretation")

eff_pct = prediction * 100
if prediction < 0.04:
    st.warning("Very low efficiency (<4%) → Poor reservoir — not recommended")
    eff_label = "Very low efficiency - Poor reservoir"
    eff_color = colors.HexColor("#c0392b")
elif prediction < 0.08:
    st.warning("Low efficiency (4-8%) → Marginal reservoir")
    eff_label = "Low efficiency - Marginal reservoir"
    eff_color = colors.HexColor("#e67e22")
elif prediction < 0.12:
    st.info("Moderate efficiency (8-12%) → Acceptable reservoir")
    eff_label = "Moderate efficiency - Acceptable reservoir"
    eff_color = colors.HexColor("#f39c12")
elif prediction < 0.16:
    st.success("Good efficiency (12-16%) → Suitable reservoir")
    eff_label = "Good efficiency - Suitable reservoir"
    eff_color = colors.HexColor("#27ae60")
else:
    st.success("High efficiency (>16%) → Excellent reservoir")
    eff_label = "High efficiency - Excellent reservoir"
    eff_color = colors.HexColor("#1a8a4a")

st.caption("Efficiency scale based on USGS/DOE open aquifer benchmarks (Bachu 2015, Celia 2015): typical range 1–20%")

# -----------------------------
# SENSITIVITY ANALYSIS
# -----------------------------
st.write("## 📊 Sensitivity Analysis")

base_input = np.array([[porosity_in, pressure_in, temperature_in, depth_in, sgr_in]])
base_pred  = float(model.predict(base_input)[0])

results = []
params  = ['Porosity', 'Pressure', 'Temperature', 'Depth', 'Residual_Gas_Saturation']

for i, param in enumerate(params):
    temp_vals    = [porosity_in, pressure_in, temperature_in, depth_in, sgr_in]
    temp_vals[i] *= 1.1
    new_pred     = float(model.predict(np.array([temp_vals]))[0])
    change       = ((new_pred - base_pred) / abs(base_pred)) * 100
    results.append([param, round(new_pred * 100, 3), round(change, 2)])

sens_df = pd.DataFrame(results, columns=["Parameter", "New Efficiency (%)", "% Change"])
sens_df["Parameter"] = sens_df["Parameter"].replace({"Residual_Gas_Saturation": "Sgr"})

st.dataframe(sens_df)

fig, ax = plt.subplots(figsize=(8, 4))
bar_colors = ["#e74c3c" if v < 0 else "#2e86c1" for v in sens_df["% Change"]]
ax.bar(sens_df["Parameter"], sens_df["% Change"], color=bar_colors)
ax.set_ylabel("% Change in Efficiency")
ax.set_title("Sensitivity Impact")
ax.axhline(0, color='black', linewidth=0.8)
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
ax.grid(True, axis='y', linestyle='--', alpha=0.6)
fig.savefig("sensitivity.png", dpi=150)
st.pyplot(fig)

# -----------------------------
# PARAMETER RANKING
# -----------------------------
st.write("## 🏆 Parameter Importance Ranking")

sens_df["Impact Strength"] = sens_df["% Change"].abs()
rank_df = sens_df.sort_values(by="Impact Strength", ascending=False)

st.dataframe(rank_df[["Parameter", "% Change"]])
st.success(f"Most Influential Parameter: {rank_df.iloc[0]['Parameter']}")

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.bar(rank_df["Parameter"], rank_df["Impact Strength"], color="#2e86c1")
ax2.set_ylabel("Impact Strength (%)")
ax2.set_title("Parameter Ranking")
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
fig2.savefig("ranking.png", dpi=150)
st.pyplot(fig2)

# -----------------------------
# PDF GENERATION
# -----------------------------
def generate_pdf():
    doc = SimpleDocTemplate(
        "CO2_Report.pdf",
        pagesize=A4,
        topMargin=0.6*inch,
        bottomMargin=0.6*inch,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("ReportTitle", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=20,
        textColor=colors.HexColor("#1a5276"), spaceAfter=4, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"],
        fontName="Helvetica", fontSize=11,
        textColor=colors.HexColor("#5d6d7e"), spaceAfter=12, alignment=TA_CENTER)
    section_style = ParagraphStyle("SectionHead", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=13,
        textColor=colors.HexColor("#1a5276"), spaceBefore=14, spaceAfter=6)
    note_style = ParagraphStyle("Note", parent=styles["Normal"],
        fontName="Helvetica-Oblique", fontSize=9,
        textColor=colors.HexColor("#7f8c8d"), spaceAfter=4)
    footer_style = ParagraphStyle("Footer", parent=styles["Normal"],
        fontName="Helvetica", fontSize=8,
        textColor=colors.HexColor("#aab7b8"), alignment=TA_CENTER)

    story = []

    # HEADER
    story.append(Paragraph("CO<sub>2</sub> Storage Prediction Report", title_style))
    story.append(Paragraph("Data-Driven Reservoir Evaluation System", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a5276"), spaceAfter=12))

    # INPUT PARAMETERS
    story.append(Paragraph("Input Parameters", section_style))
    param_data = [
        ["Parameter",           "Value",                          "Parameter",        "Value"],
        ["Porosity",            f"{round(porosity_in, 4)}",       "Pressure (psi)",   f"{pressure_in}"],
        ["Temperature (°C)",    f"{temperature_in}",              "Depth (m)",        f"{depth_in}"],
        ["Residual Gas Sat.",   f"{round(sgr_in, 3)}",            "Thickness (m)",    f"{thickness_in}"],
        ["Area (km²)",          f"{area_in}",                     "",                 ""],
    ]
    col_w = [1.5*inch, 1.2*inch, 1.5*inch, 1.2*inch]
    pt = RLTable(param_data, colWidths=col_w)
    pt.setStyle(TableStyle([
        ("BACKGROUND",     (0,0),(-1,0),  colors.HexColor("#1a5276")),
        ("TEXTCOLOR",      (0,0),(-1,0),  colors.white),
        ("FONTNAME",       (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0,0),(-1,0),  10),
        ("FONTNAME",       (0,1),(-1,-1), "Helvetica"),
        ("FONTSIZE",       (0,1),(-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1),(-1,-1), [colors.HexColor("#eaf4fb"), colors.white]),
        ("GRID",           (0,0),(-1,-1), 0.5, colors.HexColor("#aed6f1")),
        ("PADDING",        (0,0),(-1,-1), 6),
        ("ALIGN",          (1,0),(1,-1),  "CENTER"),
        ("ALIGN",          (3,0),(3,-1),  "CENTER"),
    ]))
    story.append(pt)
    story.append(Spacer(1, 10))

    # RESULTS
    story.append(Paragraph("Prediction Results", section_style))
    results_data = [
        ["Metric",                             "Value"],
        ["CO2 Storage Efficiency",             f"{round(prediction * 100, 2)}%"],
        ["Constrained Capacity (tonnes)",      f"{round(capacity_tonnes, 0):,.0f}"],
        ["Theoretical Max (tonnes)",           f"{round(theoretical_capacity, 0):,.0f}"],
        ["Operational Reduction",              f"{reduction_pct}%"],
        ["Reservoir Classification",           eff_label],
    ]
    rt = RLTable(results_data, colWidths=[3.2*inch, 3.2*inch])
    rt.setStyle(TableStyle([
        ("BACKGROUND",     (0,0),(-1,0),  colors.HexColor("#1a5276")),
        ("TEXTCOLOR",      (0,0),(-1,0),  colors.white),
        ("FONTNAME",       (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0,0),(-1,0),  10),
        ("FONTNAME",       (0,1),(-1,-1), "Helvetica"),
        ("FONTSIZE",       (0,1),(-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1),(-1,-1), [colors.HexColor("#eaf4fb"), colors.white]),
        ("GRID",           (0,0),(-1,-1), 0.5, colors.HexColor("#aed6f1")),
        ("PADDING",        (0,0),(-1,-1), 7),
        ("TEXTCOLOR",      (1,5),(1,5),   eff_color),
        ("FONTNAME",       (1,5),(1,5),   "Helvetica-Bold"),
    ]))
    story.append(rt)
    story.append(Spacer(1, 10))

    # CONSTRAINT FACTORS
    story.append(Paragraph("Capacity Constraint Factors", section_style))
    story.append(Paragraph(
        "Based on DOE/USGS volumetric methodology with operational constraints (Bachu 2015).",
        note_style))
    constraint_data = [
        ["Constraint",          "Value",                               "Description"],
        ["Sweep Efficiency",    f"{round(sweep_efficiency*100,1)}%",   "% of pore volume swept (20-35% industry standard)"],
        ["Pressure Utilization",f"{round(pressure_utilization*100,1)}%","Headroom before overpressure risk"],
        ["Depth Factor",        f"{round(depth_factor*100,1)}%",       "Injectivity at reservoir depth"],
        ["Compartmentalization",f"{round(compartment_factor*100,1)}%", "Fault isolation limits effective volume"],
    ]
    ct = RLTable(constraint_data, colWidths=[1.8*inch, 0.9*inch, 3.7*inch])
    ct.setStyle(TableStyle([
        ("BACKGROUND",     (0,0),(-1,0),  colors.HexColor("#1a5276")),
        ("TEXTCOLOR",      (0,0),(-1,0),  colors.white),
        ("FONTNAME",       (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0,0),(-1,0),  10),
        ("FONTNAME",       (0,1),(-1,-1), "Helvetica"),
        ("FONTSIZE",       (0,1),(-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1),(-1,-1), [colors.HexColor("#eaf4fb"), colors.white]),
        ("GRID",           (0,0),(-1,-1), 0.5, colors.HexColor("#aed6f1")),
        ("PADDING",        (0,0),(-1,-1), 7),
        ("ALIGN",          (1,0),(1,-1),  "CENTER"),
    ]))
    story.append(ct)
    story.append(Spacer(1, 12))

    # CHARTS
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#aed6f1"), spaceAfter=10))
    story.append(Paragraph("Analysis Charts", section_style))
    chart_row = RLTable([[
        Image("sensitivity.png", width=3.1*inch, height=2.2*inch),
        Image("ranking.png",     width=3.1*inch, height=2.2*inch),
    ]], colWidths=[3.3*inch, 3.3*inch])
    chart_row.setStyle(TableStyle([
        ("ALIGN",  (0,0),(-1,-1), "CENTER"),
        ("VALIGN", (0,0),(-1,-1), "MIDDLE"),
        ("PADDING",(0,0),(-1,-1), 4),
    ]))
  
