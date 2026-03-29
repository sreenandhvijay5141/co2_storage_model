import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

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

required_columns = [
    'Porosity', 'Pressure', 'Temperature',
    'Depth', 'Residual_Gas_Saturation', 'Efficiency'
]

if data_option == "Upload Real Dataset":
    file = st.file_uploader("Upload CSV", type=["csv"])
    
    if file is not None:
        df = pd.read_csv(file)

        if all(col in df.columns for col in required_columns):
            st.success("Dataset uploaded successfully")
        else:
            st.error("Dataset missing required columns")
            st.stop()
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
# MODEL
# -----------------------------
X = df[['Porosity', 'Pressure', 'Temperature', 'Depth', 'Residual_Gas_Saturation']]
y = df['Efficiency']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# INPUTS
# -----------------------------
st.sidebar.header("🔧 Input Parameters")

porosity = st.sidebar.slider("Porosity", float(X['Porosity'].min()), float(X['Porosity'].max()), float(X['Porosity'].mean()))
pressure = st.sidebar.slider("Pressure", int(X['Pressure'].min()), int(X['Pressure'].max()), int(X['Pressure'].mean()))
temperature = st.sidebar.slider("Temperature", int(X['Temperature'].min()), int(X['Temperature'].max()), int(X['Temperature'].mean()))
depth = st.sidebar.slider("Depth", int(X['Depth'].min()), int(X['Depth'].max()), int(X['Depth'].mean()))
sgr = st.sidebar.slider("Residual Gas Saturation", float(X['Residual_Gas_Saturation'].min()), float(X['Residual_Gas_Saturation'].max()), float(X['Residual_Gas_Saturation'].mean()))

# -----------------------------
# PREDICTION
# -----------------------------
input_data = np.array([[porosity, pressure, temperature, depth, sgr]])
prediction = model.predict(input_data)[0]

st.write("## 🎯 Prediction")
st.metric("CO₂ Storage Efficiency", round(prediction,3))

# -----------------------------
# INTERPRETATION
# -----------------------------
if prediction < 0.3:
    interpretation = "Low efficiency → Poor reservoir"
    st.warning(interpretation)
elif prediction < 0.5:
    interpretation = "Moderate efficiency → Acceptable reservoir"
    st.info(interpretation)
else:
    interpretation = "High efficiency → Excellent reservoir"
    st.success(interpretation)

# -----------------------------
# SENSITIVITY
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

# -----------------------------
# PDF GENERATION
# -----------------------------
st.write("## 📄 Generate Report")

def create_pdf():
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(sens_df["Parameter"], sens_df["% Change"])
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig("chart.png")
    plt.close()

    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("CO₂ Storage Efficiency Report", styles['Title']))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Predicted Efficiency: {round(prediction,3)}", styles['Normal']))
    content.append(Paragraph(f"Interpretation: {interpretation}", styles['Normal']))

    content.append(Spacer(1, 10))

    content.append(Paragraph("Input Parameters:", styles['Heading2']))
    content.append(Paragraph(f"Porosity: {porosity}", styles['Normal']))
    content.append(Paragraph(f"Pressure: {pressure}", styles['Normal']))
    content.append(Paragraph(f"Temperature: {temperature}", styles['Normal']))
    content.append(Paragraph(f"Depth: {depth}", styles['Normal']))
    content.append(Paragraph(f"Sgr: {sgr}", styles['Normal']))

    content.append(Spacer(1, 10))
    content.append(Paragraph("Sensitivity Analysis Chart:", styles['Heading2']))
    content.append(Image("chart.png", width=400, height=200))

    doc.build(content)

    with open("report.pdf", "rb") as f:
        return f.read()

pdf = create_pdf()

st.download_button("📥 Download PDF Report", pdf, "CO2_Report.pdf")
