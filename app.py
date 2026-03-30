import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# PDF
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
# DATASET
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

    df = pd.DataFrame({
        'Porosity': np.random.uniform(0.1, 0.3, n),
        'Pressure': np.random.uniform(2000, 5000, n),
        'Temperature': np.random.uniform(50, 100, n),
        'Depth': np.random.uniform(1500, 3500, n),
        'Residual_Gas_Saturation': np.random.uniform(0.1, 0.5, n),
    })

    df['Efficiency'] = (
        0.6 * df['Porosity']**2 +
        0.00004 * df['Pressure'] +
        0.00008 * df['Depth'] -
        0.002 * df['Temperature'] +
        0.5 * df['Residual_Gas_Saturation'] +
        np.random.normal(0, 0.02, n)
    )

# -----------------------------
# MODEL
# -----------------------------
X = df[['Porosity', 'Pressure', 'Temperature', 'Depth', 'Residual_Gas_Saturation']]
y = df['Efficiency']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("🔧 Input Parameters")

porosity = st.sidebar.slider("Porosity", float(X['Porosity'].min()), float(X['Porosity'].max()), float(X['Porosity'].mean()))
pressure = st.sidebar.slider("Pressure", int(X['Pressure'].min()), int(X['Pressure'].max()), int(X['Pressure'].mean()))
temperature = st.sidebar.slider("Temperature", int(X['Temperature'].min()), int(X['Temperature'].max()), int(X['Temperature'].mean()))
depth = st.sidebar.slider("Depth", int(X['Depth'].min()), int(X['Depth'].max()), int(X['Depth'].mean()))
sgr = st.sidebar.slider("Residual Gas Saturation", float(X['Residual_Gas_Saturation'].min()), float(X['Residual_Gas_Saturation'].max()), float(X['Residual_Gas_Saturation'].mean()))

# ADDED
thickness = st.sidebar.slider("Reservoir Thickness (m)", 10, 200, 50)
area = st.sidebar.slider("Reservoir Area (km²)", 1, 500, 50)

# -----------------------------
# PERFORMANCE
# -----------------------------
r2 = model.score(X_test, y_test)
rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

st.write("## 📊 Model Performance")
col1, col2 = st.columns(2)
col1.metric("R² Score", round(r2,3))
col2.metric("RMSE", round(rmse,4))

# -----------------------------
# PREDICTION
# -----------------------------
input_data = np.array([[porosity, pressure, temperature, depth, sgr]])
prediction = model.predict(input_data)[0]

# CO2 Density (approx)
co2_density = 600 + (pressure / 100) - (temperature * 2)
co2_density = max(300, min(co2_density, 800))

# Capacity
area_m2 = area * 1e6
efficiency_factor = (1 - sgr)

capacity_kg = area_m2 * thickness * porosity * co2_density * efficiency_factor
capacity_tonnes = capacity_kg / 1000

st.write("## 🎯 Prediction")
st.metric("CO₂ Storage Efficiency", round(prediction,3))
st.metric("CO₂ Storage Capacity (tonnes)", round(capacity_tonnes, 2))

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
base_pred = model.predict(base_input)[0]

results = []
params = ['Porosity','Pressure','Temperature','Depth','Residual_Gas_Saturation']

for i, param in enumerate(params):
    temp = [porosity, pressure, temperature, depth, sgr]
    temp[i] *= 1.1
    temp = np.array([temp])
    new_pred = model.predict(temp)[0]
    change = ((new_pred - base_pred)/base_pred)*100
    results.append([param, round(new_pred,3), round(change,2)])

sens_df = pd.DataFrame(results, columns=["Parameter", "New Prediction", "% Change"])
sens_df["Parameter"] = sens_df["Parameter"].replace({"Residual_Gas_Saturation": "Sgr"})

st.dataframe(sens_df)

fig, ax = plt.subplots(figsize=(8,4))
ax.bar(sens_df["Parameter"], sens_df["% Change"])
ax.set_ylabel("% Change in Efficiency")
ax.set_title("Sensitivity Impact")
plt.xticks(rotation=25)
plt.tight_layout()
fig.savefig("sensitivity.png")
st.pyplot(fig)

# -----------------------------
# RANKING
# -----------------------------
st.write("## 🏆 Parameter Importance Ranking")

sens_df["Impact Strength"] = sens_df["% Change"].abs()
rank_df = sens_df.sort_values(by="Impact Strength", ascending=False)

st.dataframe(rank_df[["Parameter", "% Change"]])

fig2, ax2 = plt.subplots(figsize=(8,4))
ax2.bar(rank_df["Parameter"], rank_df["Impact Strength"])
ax2.set_title("Parameter Ranking")
plt.xticks(rotation=25)
plt.tight_layout()
fig2.savefig("ranking.png")
st.pyplot(fig2)

# -----------------------------
# PDF FUNCTION
# -----------------------------
def generate_pdf():
    doc = SimpleDocTemplate("report.pdf", pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>CO₂ Storage Report</b>", styles["Title"]))
    story.append(Spacer(1,12))

    story.append(Paragraph("<b>Inputs:</b>", styles["Normal"]))
    story.append(Paragraph(f"Porosity: {porosity}", styles["Normal"]))
    story.append(Paragraph(f"Pressure: {pressure}", styles["Normal"]))
    story.append(Paragraph(f"Temperature: {temperature}", styles["Normal"]))
    story.append(Paragraph(f"Depth: {depth}", styles["Normal"]))
    story.append(Paragraph(f"Sgr: {sgr}", styles["Normal"]))
    story.append(Paragraph(f"Thickness: {thickness}", styles["Normal"]))
    story.append(Paragraph(f"Area: {area}", styles["Normal"]))

    story.append(Spacer(1,12))

    story.append(Paragraph("<b>Results:</b>", styles["Normal"]))
    story.append(Paragraph(f"Efficiency: {round(prediction,3)}", styles["Normal"]))
    story.append(Paragraph(f"Capacity: {round(capacity_tonnes,2)} tonnes", styles["Normal"]))

    story.append(Spacer(1,12))
    story.append(Image("sensitivity.png", width=400, height=200))
    story.append(Spacer(1,12))
    story.append(Image("ranking.png", width=400, height=200))

    doc.build(story)

    with open("report.pdf", "rb") as f:
        return f.read()

# -----------------------------
# DOWNLOAD
# -----------------------------
st.write("## 📥 Download Result")

st.download_button("Download CSV",
                   pd.DataFrame({"Efficiency":[prediction]}).to_csv(index=False),
                   "result.csv")

pdf = generate_pdf()

st.download_button("Download PDF Report", pdf, "report.pdf")
