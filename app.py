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
# SIDEBAR
# -----------------------------
st.sidebar.header("🔧 Input Parameters")

porosity = st.sidebar.slider("Porosity", float(X['Porosity'].min()), float(X['Porosity'].max()), float(X['Porosity'].mean()))
pressure = st.sidebar.slider("Pressure", int(X['Pressure'].min()), int(X['Pressure'].max()), int(X['Pressure'].mean()))
temperature = st.sidebar.slider("Temperature", int(X['Temperature'].min()), int(X['Temperature'].max()), int(X['Temperature'].mean()))
depth = st.sidebar.slider("Depth", int(X['Depth'].min()), int(X['Depth'].max()), int(X['Depth'].mean()))
sgr = st.sidebar.slider("Residual Gas Saturation", float(X['Residual_Gas_Saturation'].min()), float(X['Residual_Gas_Saturation'].max()), float(X['Residual_Gas_Saturation'].mean()))

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

# -----------------------------
# FIXED CAPACITY (REAL WORLD)
# -----------------------------
area_m2 = area * 1e6

max_pressure = 5000
pressure_utilization = 1 - (pressure / max_pressure) * 0.6
pressure_utilization = max(0.1, min(pressure_utilization, 0.9))

depth_factor = min(1.0, 800 / max(depth, 500))
compartment_factor = max(0.05, 1 - (depth / 10000))

co2_density = 600 + (pressure / 100) - (temperature * 2)
co2_density = max(300, min(co2_density, 800))

efficiency_factor = (1 - sgr)

capacity_kg = (
    area_m2 * thickness * porosity * co2_density
    * efficiency_factor
    * pressure_utilization
    * depth_factor
    * compartment_factor
)

capacity_tonnes = capacity_kg / 1000

st.write("## 🎯 Prediction")
st.metric("CO₂ Storage Efficiency", round(prediction,3))
st.metric("CO₂ Storage Capacity (tonnes)", round(capacity_tonnes, 2))

# -----------------------------
# SENSITIVITY ANALYSIS
# -----------------------------
st.write("## 📊 Sensitivity Analysis")

base_capacity = capacity_kg
results = []

params = ['Porosity','Pressure','Temperature','Depth','Residual_Gas_Saturation','Thickness','Area']

for param in params:
    tp, pr, te, de, sg = porosity, pressure, temperature, depth, sgr
    th, ar = thickness, area

    if param == 'Porosity': tp *= 1.1
    elif param == 'Pressure': pr *= 1.1
    elif param == 'Temperature': te *= 1.1
    elif param == 'Depth': de *= 1.1
    elif param == 'Residual_Gas_Saturation': sg *= 1.1
    elif param == 'Thickness': th *= 1.1
    elif param == 'Area': ar *= 1.1

    temp_area_m2 = ar * 1e6

    temp_pressure_utilization = 1 - (pr / max_pressure) * 0.6
    temp_pressure_utilization = max(0.1, min(temp_pressure_utilization, 0.9))

    temp_depth_factor = min(1.0, 800 / max(de, 500))
    temp_compartment_factor = max(0.05, 1 - (de / 10000))

    temp_density = 600 + (pr / 100) - (te * 2)
    temp_density = max(300, min(temp_density, 800))

    temp_eff = (1 - sg)

    new_capacity = (
        temp_area_m2 * th * tp * temp_density
        * temp_eff
        * temp_pressure_utilization
        * temp_depth_factor
        * temp_compartment_factor
    )

    change = ((new_capacity - base_capacity) / base_capacity) * 100

    results.append([param, round(change,2)])

sens_df = pd.DataFrame(results, columns=["Parameter", "% Change"])

sens_df["Parameter"] = sens_df["Parameter"].replace({
    "Residual_Gas_Saturation": "Sgr"
})

st.dataframe(sens_df)

fig, ax = plt.subplots()
ax.bar(sens_df["Parameter"], sens_df["% Change"])
plt.xticks(rotation=25)
st.pyplot(fig)

# -----------------------------
# PARAMETER RANKING
# -----------------------------
st.write("## 🏆 Parameter Importance Ranking")

sens_df["Impact Strength"] = sens_df["% Change"].abs()
rank_df = sens_df.sort_values(by="Impact Strength", ascending=False)

st.dataframe(rank_df)

top_param = rank_df.iloc[0]["Parameter"]
st.success(f"Most Influential Parameter: {top_param}")
