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
    ["Synthetic Dataset", "Upload Real Dataset"]
)

if data_option == "Upload Real Dataset":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.success("Dataset uploaded successfully")
        st.dataframe(df.head())
    else:
        st.stop()

else:
    np.random.seed(42)
    n = 120

    data = {
        'Porosity': np.random.uniform(0.1, 0.3, n),
        'Pressure': np.random.uniform(2000, 5000, n),
        'Temperature': np.random.uniform(50, 100, n),
        'Depth': np.random.uniform(1500, 3500, n),
        'Residual_Gas_Saturation': np.random.uniform(0.1, 0.5, n),
    }

    df = pd.DataFrame(data)

    df['Efficiency'] = (
        0.6 * df['Porosity']**2 +
        0.00004 * df['Pressure'] +
        0.00008 * df['Depth'] -
        0.002 * df['Temperature'] +
        0.5 * df['Residual_Gas_Saturation'] +
        np.random.normal(0, 0.02, n)
    )

# -----------------------------
# FEATURES
# -----------------------------
X = df[['Porosity', 'Pressure', 'Temperature', 'Depth', 'Residual_Gas_Saturation']]
y = df['Efficiency']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("🔧 Input Parameters")

porosity    = st.sidebar.slider("Porosity",                  float(X['Porosity'].min()),                  float(X['Porosity'].max()),                  float(X['Porosity'].mean()))
pressure    = st.sidebar.slider("Pressure",                  int(X['Pressure'].min()),                    int(X['Pressure'].max()),                    int(X['Pressure'].mean()))
temperature = st.sidebar.slider("Temperature",               int(X['Temperature'].min()),                 int(X['Temperature'].max()),                 int(X['Temperature'].mean()))
depth       = st.sidebar.slider("Depth",                     int(X['Depth'].min()),                       int(X['Depth'].max()),                       int(X['Depth'].mean()))
sgr         = st.sidebar.slider("Residual Gas Saturation",   float(X['Residual_Gas_Saturation'].min()),   float(X['Residual_Gas_Saturation'].max()),   float(X['Residual_Gas_Saturation'].mean()))
thickness   = st.sidebar.slider("Reservoir Thickness (m)",   10, 200, 50)
area        = st.sidebar.slider("Reservoir Area (km²)",      1, 500, 50)

# -----------------------------
# MODEL
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

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
input_data = np.array([[porosity, pressure, temperature, depth, sgr]])
prediction = model.predict(input_data)[0]

# -----------------------------
# CAPACITY CALCULATION (with real-world constraints)
# -----------------------------
area_m2 = area * 1e6

# CO2 density approximation (supercritical state)
co2_density = 600 + (pressure / 100) - (temperature * 2)
co2_density = max(300, min(co2_density, 800))

# 1. Pressure utilization — high pressure = less room before overpressure risk
max_pressure = 5000
pressure_utilization = 1 - (pressure / max_pressure) * 0.6
pressure_utilization = max(0.1, min(pressure_utilization, 0.9))

# 2. Depth factor — diminishing injectivity returns beyond ~3000m
depth_factor = min(1.0, 800 / max(depth, 500))

# 3. Compartmentalization — fault isolation limits effective reservoir volume
compartment_factor = max(0.05, 1 - (depth / 10000))

# 4. Residual gas efficiency
efficiency_factor = (1 - sgr)

# 5. Theoretical max (no constraints)
theoretical_capacity = (area_m2 * thickness * porosity * co2_density * efficiency_factor) / 1000

# 6. Constrained capacity
capacity_kg = (
    area_m2 * thickness * porosity * co2_density
    * efficiency_factor
    * pressure_utilization
    * depth_factor
    * compartment_factor
)
capacity_tonnes = capacity_kg / 1000

# -----------------------------
# DISPLAY PREDICTION
# -----------------------------
st.write("## 🎯 Prediction")
col1, col2 = st.columns(2)
col1.metric("CO₂ Storage Efficiency", round(prediction, 3))
col2.metric("CO₂ Storage Capacity (tonnes)", f"{round(capacity_tonnes, 0):,.0f}")

# -----------------------------
# CAPACITY CONSTRAINT BREAKDOWN
# -----------------------------
st.write("## 🔍 Capacity Constraint Breakdown")
st.caption("Shows how each real-world factor reduces the theoretical maximum capacity.")

col1, col2, col3 = st.columns(3)
col1.metric("Pressure Utilization", f"{round(pressure_utilization * 100, 1)}%")
col2.metric("Depth Factor",         f"{round(depth_factor * 100, 1)}%")
col3.metric("Compartmentalization", f"{round(compartment_factor * 100, 1)}%")

reduction_pct = round((1 - capacity_tonnes / theoretical_capacity) * 100, 1)
st.info(
    f"📌 Theoretical max (no constraints): **{round(theoretical_capacity, 0):,.0f} tonnes**  \n"
    f"✅ Constrained estimate: **{round(capacity_tonnes, 0):,.0f} tonnes**  \n"
    f"📉 Reduction applied: **{reduction_pct}%**"
)

# -----------------------------
# INTERPRETATION
# -----------------------------
st.write("## 📘 Interpretation")

if prediction < 0.25:
    st.warning("Low efficiency → Poor reservoir")
    eff_label = "Low efficiency - Poor reservoir"
    eff_color = colors.HexColor("#e74c3c")
elif prediction < 0.40:
    st.info("Moderate efficiency → Acceptable reservoir")
    eff_label = "Moderate efficiency - Acceptable reservoir"
    eff_color = colors.HexColor("#f39c12")
elif prediction < 0.60:
    st.success("Good efficiency → Suitable reservoir")
    eff_label = "Good efficiency - Suitable reservoir"
    eff_color = colors.HexColor("#27ae60")
else:
    st.success("High efficiency → Excellent reservoir")
    eff_label = "High efficiency - Excellent reservoir"
    eff_color = colors.HexColor("#1a8a4a")

# -----------------------------
# SENSITIVITY ANALYSIS
# -----------------------------
st.write("## 📊 Sensitivity Analysis")

base_input = np.array([[porosity, pressure, temperature, depth, sgr]])
base_pred  = model.predict(base_input)[0]

results = []
params  = ['Porosity', 'Pressure', 'Temperature', 'Depth', 'Residual_Gas_Saturation']

for i, param in enumerate(params):
    temp_vals = [porosity, pressure, temperature, depth, sgr]
    temp_vals[i] *= 1.1
    new_pred = model.predict(np.array([temp_vals]))[0]
    change   = ((new_pred - base_pred) / base_pred) * 100
    results.append([param, round(new_pred, 3), round(change, 2)])

sens_df = pd.DataFrame(results, columns=["Parameter", "New Prediction", "% Change"])
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

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=colors.HexColor("#1a5276"),
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        textColor=colors.HexColor("#5d6d7e"),
        spaceAfter=12,
        alignment=TA_CENTER,
    )
    section_style = ParagraphStyle(
        "SectionHead",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=colors.HexColor("#1a5276"),
        spaceBefore=14,
        spaceAfter=6,
    )
    note_style = ParagraphStyle(
        "Note",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        textColor=colors.HexColor("#7f8c8d"),
        spaceAfter=4,
    )

    story = []

    # ── HEADER ──────────────────────────────────────────────────────
    story.append(Paragraph("CO<sub>2</sub> Storage Prediction Report", title_style))
    story.append(Paragraph("Data-Driven Reservoir Evaluation System", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a5276"), spaceAfter=12))

    # ── INPUT PARAMETERS ────────────────────────────────────────────
    story.append(Paragraph("Input Parameters", section_style))

    param_data = [
        ["Parameter",           "Value",                    "Parameter",         "Value"],
        ["Porosity",            f"{round(porosity, 4)}",    "Pressure (psi)",    f"{pressure}"],
        ["Temperature (°C)",    f"{temperature}",           "Depth (m)",         f"{depth}"],
        ["Residual Gas Sat.",   f"{round(sgr, 3)}",         "Thickness (m)",     f"{thickness}"],
        ["Area (km²)",          f"{area}",                  "",                  ""],
    ]

    col_w = [1.5*inch, 1.2*inch, 1.5*inch, 1.2*inch]
    param_table = RLTable(param_data, colWidths=col_w)
    param_table.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  colors.HexColor("#1a5276")),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, 0),  10),
        ("FONTNAME",       (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#eaf4fb"), colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#aed6f1")),
        ("PADDING",        (0, 0), (-1, -1), 6),
        ("ALIGN",          (1, 0), (1, -1),  "CENTER"),
        ("ALIGN",          (3, 0), (3, -1),  "CENTER"),
    ]))
    story.append(param_table)
    story.append(Spacer(1, 10))

    # ── RESULTS ─────────────────────────────────────────────────────
    story.append(Paragraph("Prediction Results", section_style))

    results_data = [
        ["Metric",                              "Value"],
        ["CO2 Storage Efficiency",              f"{round(prediction, 3)}"],
        ["Constrained Capacity (tonnes)",       f"{round(capacity_tonnes, 0):,.0f}"],
        ["Theoretical Max Capacity (tonnes)",   f"{round(theoretical_capacity, 0):,.0f}"],
        ["Capacity Reduction",                  f"{reduction_pct}%"],
        ["Reservoir Classification",            eff_label],
    ]

    res_table = RLTable(results_data, colWidths=[3.2*inch, 3.2*inch])
    res_table.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  colors.HexColor("#1a5276")),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, 0),  10),
        ("FONTNAME",       (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#eaf4fb"), colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#aed6f1")),
        ("PADDING",        (0, 0), (-1, -1), 7),
        ("TEXTCOLOR",      (1, 5), (1, 5),   eff_color),
        ("FONTNAME",       (1, 5), (1, 5),   "Helvetica-Bold"),
    ]))
    story.append(res_table)
    story.append(Spacer(1, 10))

    # ── CONSTRAINT FACTORS ──────────────────────────────────────────
    story.append(Paragraph("Capacity Constraint Factors", section_style))
    story.append(Paragraph(
        "These factors reduce the theoretical maximum to a realistic constrained estimate.",
        note_style
    ))

    constraint_data = [
        ["Constraint",           "Value",                                "Description"],
        ["Pressure Utilization", f"{round(pressure_utilization*100,1)}%", "High pressure limits safe injection volume"],
        ["Depth Factor",         f"{round(depth_factor*100,1)}%",         "Deep reservoirs have reduced injectivity"],
        ["Compartmentalization", f"{round(compartment_factor*100,1)}%",   "Fault isolation limits effective volume"],
    ]

    con_table = RLTable(constraint_data, colWidths=[1.8*inch, 0.9*inch, 3.7*inch])
    con_table.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  colors.HexColor("#1a5276")),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, 0),  10),
        ("FONTNAME",       (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#eaf4fb"), colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#aed6f1")),
        ("PADDING",        (0, 0), (-1, -1), 7),
        ("ALIGN",          (1, 0), (1, -1),  "CENTER"),
    ]))
    story.append(con_table)
    story.append(Spacer(1, 12))

    # ── CHARTS SIDE BY SIDE ─────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#aed6f1"), spaceAfter=10))
    story.append(Paragraph("Analysis Charts", section_style))

    chart_row = RLTable(
        [[
            Image("sensitivity.png", width=3.1*inch, height=2.2*inch),
            Image("ranking.png",     width=3.1*inch, height=2.2*inch),
        ]],
        colWidths=[3.3*inch, 3.3*inch]
    )
    chart_row.setStyle(TableStyle([
        ("ALIGN",   (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",  (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(chart_row)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Left: Sensitivity impact of each parameter on storage efficiency.  "
        "Right: Parameters ranked by absolute impact strength.",
        note_style
    ))

    # ── FOOTER ──────────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#aed6f1"), spaceAfter=4))
    story.append(Paragraph(
        "Generated by CO<sub>2</sub> Storage Prediction System  |  Data-Driven Reservoir Evaluation",
        ParagraphStyle(
            "Footer", parent=styles["Normal"],
            fontName="Helvetica", fontSize=8,
            textColor=colors.HexColor("#aab7b8"),
            alignment=TA_CENTER
        )
    ))

    doc.build(story)
    with open("CO2_Report.pdf", "rb") as f:
        return f.read()

# -----------------------------
# DOWNLOAD
# -----------------------------
st.write("## 📥 Download Result")

output_df = pd.DataFrame({
    "Porosity":                         [porosity],
    "Pressure":                         [pressure],
    "Temperature":                      [temperature],
    "Depth":                            [depth],
    "Residual Gas Saturation":          [sgr],
    "Thickness (m)":                    [thickness],
    "Area (km2)":                       [area],
    "Predicted Efficiency":             [round(prediction, 3)],
    "CO2 Capacity Constrained (t)":     [round(capacity_tonnes, 0)],
    "CO2 Capacity Theoretical (t)":     [round(theoretical_capacity, 0)],
    "Pressure Utilization (%)":         [round(pressure_utilization * 100, 1)],
    "Depth Factor (%)":                 [round(depth_factor * 100, 1)],
    "Compartmentalization (%)":         [round(compartment_factor * 100, 1)],
})

st.download_button("⬇️ Download CSV", output_df.to_csv(index=False), "result.csv")

pdf_data = generate_pdf()
st.download_button(
    label="⬇️ Download PDF Report",
    data=pdf_data,
    file_name="CO2_Report.pdf",
    mime="application/pdf"
)
