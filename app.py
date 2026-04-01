import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="CO₂ Storage Model", layout="wide")

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("🔧 Input Parameters")

porosity    = st.sidebar.slider("Porosity", float(X['Porosity'].min()), float(X['Porosity'].max()), float(X['Porosity'].mean()))
pressure    = st.sidebar.slider("Pressure", int(X['Pressure'].min()), int(X['Pressure'].max()), int(X['Pressure'].mean()))
temperature = st.sidebar.slider("Temperature", int(X['Temperature'].min()), int(X['Temperature'].max()), int(X['Temperature'].mean()))
depth       = st.sidebar.slider("Depth", int(X['Depth'].min()), int(X['Depth'].max()), int(X['Depth'].mean()))
sgr         = st.sidebar.slider("Residual Gas Saturation", float(X['Residual_Gas_Saturation'].min()), float(X['Residual_Gas_Saturation'].max()), float(X['Residual_Gas_Saturation'].mean()))
thickness   = st.sidebar.slider("Reservoir Thickness (m)", 10, 200, 50)
area        = st.sidebar.slider("Reservoir Area (km²)", 1, 500, 50)

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
# ✅ FIXED: CAPACITY CALCULATION
# -----------------------------
area_m2 = area * 1e6

# CO2 density (supercritical state approximation)
co2_density = 600 + (pressure / 100) - (temperature * 2)
co2_density = max(300, min(co2_density, 800))

# 1. Pressure utilization — high pressure = less room before overpressure risk
max_pressure = 5000  # psi (slider max)
pressure_utilization = 1 - (pressure / max_pressure) * 0.6
pressure_utilization = max(0.1, min(pressure_utilization, 0.9))

# 2. Depth factor — diminishing returns beyond ~3000m due to injection difficulty
depth_factor = min(1.0, 800 / max(depth, 500))

# 3. Compartmentalization factor — faults isolate parts of the reservoir
compartment_factor = max(0.05, 1 - (depth / 10000))

# 4. Residual gas efficiency
efficiency_factor = (1 - sgr)

# 5. Final constrained capacity
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
st.metric("CO₂ Storage Efficiency", round(prediction, 3))
st.metric("CO₂ Storage Capacity (tonnes)", round(capacity_tonnes, 2))

# -----------------------------
# ✅ NEW: CAPACITY CONSTRAINT BREAKDOWN
# -----------------------------
st.write("## 🔍 Capacity Constraint Breakdown")
st.caption("Shows how each real-world factor is reducing the theoretical maximum capacity.")

col1, col2, col3 = st.columns(3)
col1.metric("Pressure Utilization", f"{round(pressure_utilization * 100, 1)}%",
            help="High reservoir pressure limits how much CO₂ can be safely injected.")
col2.metric("Depth Factor", f"{round(depth_factor * 100, 1)}%",
            help="Very deep reservoirs have diminishing injectivity returns.")
col3.metric("Compartmentalization", f"{round(compartment_factor * 100, 1)}%",
            help="Faults isolate sections of the reservoir, reducing effective volume.")

theoretical_capacity = (area_m2 * thickness * porosity * co2_density * efficiency_factor) / 1000
st.info(
    f"📌 Theoretical max (no constraints): **{round(theoretical_capacity, 0):,.0f} tonnes**  \n"
    f"✅ Constrained estimate: **{round(capacity_tonnes, 0):,.0f} tonnes**  \n"
    f"📉 Reduction: **{round((1 - capacity_tonnes/theoretical_capacity)*100, 1)}%**"
)

# -----------------------------
# INTERPRETATION
# -----------------------------
st.write("## 📘 Interpretation")

if prediction < 0.25:
    st.warning("Low efficiency → Poor reservoir")
elif prediction < 0.40:
    st.info("Moderate efficiency → Acceptable reservoir")
elif prediction < 0.60:
    st.success("Good efficiency → Suitable reservoir")
else:
    st.success("High efficiency → Excellent reservoir")

# -----------------------------
# SENSITIVITY ANALYSIS
# -----------------------------
st.write("## 📊 Sensitivity Analysis")

base_input = np.array([[porosity, pressure, temperature, depth, sgr]])
base_pred  = model.predict(base_input)[0]

results = []
params  = ['Porosity', 'Pressure', 'Temperature', 'Depth', 'Residual_Gas_Saturation']

for i, param in enumerate(params):
    temp = [porosity, pressure, temperature, depth, sgr]
    temp[i] *= 1.1
    new_pred = model.predict(np.array([temp]))[0]
    change   = ((new_pred - base_pred) / base_pred) * 100
    results.append([param, round(new_pred, 3), round(change, 2)])

sens_df = pd.DataFrame(results, columns=["Parameter", "New Prediction", "% Change"])
sens_df["Parameter"] = sens_df["Parameter"].replace({"Residual_Gas_Saturation": "Sgr"})

st.dataframe(sens_df)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(sens_df["Parameter"], sens_df["% Change"])
ax.set_ylabel("% Change in Efficiency")
ax.set_title("Sensitivity Impact")
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
ax.grid(True, axis='y', linestyle='--', alpha=0.6)
fig.savefig("sensitivity.png")
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
ax2.bar(rank_df["Parameter"], rank_df["Impact Strength"])
ax2.set_ylabel("Impact Strength (%)")
ax2.set_title("Parameter Ranking")
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
fig2.savefig("ranking.png")
st.pyplot(fig2)

# -----------------------------
# PDF REPORT
# -----------------------------
def generate_pdf():
    doc = SimpleDocTemplate("CO2_Report.pdf", pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<font name='Times-Roman' size=16><b>CO₂ Storage Report</b></font>", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Input Parameters:</b>", styles["Normal"]))
    story.append(Paragraph(f"Porosity: {porosity}", styles["Normal"]))
    story.append(Paragraph(f"Pressure: {pressure}", styles["Normal"]))
    story.append(Paragraph(f"Temperature: {temperature}", styles["Normal"]))
    story.append(Paragraph(f"Depth: {depth}", styles["Normal"]))
    story.append(Paragraph(f"Sgr: {sgr}", styles["Normal"]))
    story.append(Paragraph(f"Thickness: {thickness}", styles["Normal"]))
    story.append(Paragraph(f"Area: {area}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Results:</b>", styles["Normal"]))
    story.append(Paragraph(f"Efficiency: {round(prediction, 3)}", styles["Normal"]))
    story.append(Paragraph(f"Capacity (Constrained): {round(capacity_tonnes, 2)} tonnes", styles["Normal"]))
    story.append(Paragraph(f"Theoretical Max: {round(theoretical_capacity, 2)} tonnes", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Constraint Factors:</b>", styles["Normal"]))
    story.append(Paragraph(f"Pressure Utilization: {round(pressure_utilization * 100, 1)}%", styles["Normal"]))
    story.append(Paragraph(f"Depth Factor: {round(depth_factor * 100, 1)}%", styles["Normal"]))
    story.append(Paragraph(f"Compartmentalization: {round(compartment_factor * 100, 1)}%", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Sensitivity Analysis:</b>", styles["Normal"]))
    story.append(Image("sensitivity.png", width=5*inch, height=3*inch))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Parameter Ranking:</b>", styles["Normal"]))
    story.append(Image("ranking.png", width=5*inch, height=3*inch))

    doc.build(story)
    with open("CO2_Report.pdf", "rb") as f:
        return f.read()

# -----------------------------
# DOWNLOAD
# -----------------------------
st.write("## 📥 Download Result")

output_df = pd.DataFrame({
    "Porosity": [porosity],
    "Pressure": [pressure],
    "Temperature": [temperature],
    "Depth": [depth],
    "Residual Gas Saturation": [sgr],
    "Thickness (m)": [thickness],
    "Area (km2)": [area],
    "Predicted Efficiency": [prediction],
    "CO2 Capacity Constrained (tonnes)": [capacity_tonnes],
    "CO2 Capacity Theoretical (tonnes)": [theoretical_capacity],
})

st.download_button("Download CSV", output_df.to_csv(index=False), "result.csv")

pdf_data = generate_pdf()
st.download_button(
    label="Download PDF Report",
    data=pdf_data,
    file_name="CO2_Report.pdf",
    mime="application/pdf"
)
