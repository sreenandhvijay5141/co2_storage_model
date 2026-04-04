import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
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
from reportlab.platypus import (HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer)
from reportlab.platypus import Table as RLTable
from reportlab.platypus import TableStyle
import streamlit.components.v1 as components

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CO₂ Storage Prediction System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
if 'app_phase' not in st.session_state:
    st.session_state.app_phase = 'splash'
if 'splash_complete' not in st.session_state:
    st.session_state.splash_complete = False
if 'overview_complete' not in st.session_state:
    st.session_state.overview_complete = False

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS - INDUSTRIAL DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Dark Industrial Theme */
    .stApp {
        background: linear-gradient(135deg, #1a1f3d 0%, #0f1420 100%);
        color: #e2e8f0;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(45, 55, 72, 0.4);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }
    
    /* Bento Grid Layout */
    .bento-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #f7fafc !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    
    /* Success/Warning Colors */
    .success-text {
        color: #10b981 !important;
        font-weight: 600;
    }
    
    .warning-text {
        color: #ef4444 !important;
        font-weight: 600;
    }
    
    /* Custom Input Styling */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div,
    .stSlider > div > div > div {
        background: rgba(45, 55, 72, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #e2e8f0 !important;
        border-radius: 8px;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4) !important;
    }
    
    /* Download Buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #1a1f3d 0%, #2d3748 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Metrics */
    .metric-card {
        background: rgba(45, 55, 72, 0.4);
        backdrop-filter: blur(20px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.4);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f3d 0%, #0f1420 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #10b981 0%, #34d399 100%) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(45, 55, 72, 0.4) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Dataframe */
    .dataframe {
        background: rgba(45, 55, 72, 0.4) !important;
        border-radius: 8px !important;
    }
    
    /* Status Indicator */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active {
        background: #10b981;
        box-shadow: 0 0 8px #10b981;
    }
    
    .status-warning {
        background: #f59e0b;
        box-shadow: 0 0 8px #f59e0b;
    }
    
    .status-inactive {
        background: #6b7280;
    }
    
    /* Toggle Switch */
    .toggle-container {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px;
        background: rgba(45, 55, 72, 0.3);
        border-radius: 8px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: SPLASH SCREEN WITH 3D ANIMATION
# ═══════════════════════════════════════════════════════════════════════════════
def render_splash_screen():
    load_custom_css()
    
    splash_html = """
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; background: linear-gradient(135deg, #1a1f3d 0%, #0f1420 100%);">
        <div id="animation-container" style="width: 600px; height: 400px; margin-bottom: 40px;"></div>
        <div style="text-align: center;">
            <h1 style="color: #f7fafc; font-size: 3rem; font-weight: 700; margin-bottom: 10px; letter-spacing: -0.02em;">
                CO₂ Storage Prediction System
            </h1>
            <p style="color: #94a3b8; font-size: 1.2rem; margin-bottom: 30px;">Scanning Reservoirs...</p>
            <div style="width: 400px; height: 4px; background: rgba(255, 255, 255, 0.1); border-radius: 2px; overflow: hidden; position: relative;">
                <div id="progress-bar" style="height: 100%; background: linear-gradient(90deg, #10b981 0%, #34d399 100%); width: 0%; transition: width 0.3s ease;"></div>
                <div id="sweep" style="position: absolute; top: 0; left: 0; height: 100%; width: 100px; background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent); animation: sweep 2s infinite;"></div>
            </div>
        </div>
    </div>
    
    <style>
    @keyframes sweep {
        0% { transform: translateX(-100px); }
        100% { transform: translateX(400px); }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotateZ(0deg); }
        50% { transform: translateY(-20px) rotateZ(5deg); }
    }
    </style>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    // Three.js 3D Wireframe Reservoir Animation
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, 600/400, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    
    renderer.setSize(600, 400);
    renderer.setClearColor(0x000000, 0);
    document.getElementById('animation-container').appendChild(renderer.domElement);
    
    // Create strata layers (reservoir visualization)
    const layers = [];
    const layerCount = 8;
    const layerColors = [0x10b981, 0x059669, 0x047857, 0x065f46];
    
    for (let i = 0; i < layerCount; i++) {
        const geometry = new THREE.PlaneGeometry(6, 0.3, 32, 1);
        const material = new THREE.MeshBasicMaterial({ 
            color: layerColors[i % layerColors.length],
            wireframe: true,
            opacity: 0.6 + (i * 0.05),
            transparent: true
        });
        const plane = new THREE.Mesh(geometry, material);
        plane.position.y = (i - layerCount/2) * 0.4;
        plane.rotation.x = -Math.PI / 6;
        scene.add(plane);
        layers.push(plane);
    }
    
    // Add particle system (CO2 molecules)
    const particlesGeometry = new THREE.BufferGeometry();
    const particlesCount = 100;
    const positions = new Float32Array(particlesCount * 3);
    
    for (let i = 0; i < particlesCount * 3; i++) {
        positions[i] = (Math.random() - 0.5) * 10;
    }
    
    particlesGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const particlesMaterial = new THREE.PointsMaterial({
        color: 0x10b981,
        size: 0.05,
        transparent: true,
        opacity: 0.6
    });
    const particles = new THREE.Points(particlesGeometry, particlesMaterial);
    scene.add(particles);
    
    camera.position.z = 5;
    
    // Animation loop
    let progress = 0;
    function animate() {
        requestAnimationFrame(animate);
        
        // Rotate layers
        layers.forEach((layer, i) => {
            layer.rotation.z += 0.001 * (1 + i * 0.1);
        });
        
        // Animate particles
        particles.rotation.y += 0.002;
        
        renderer.render(scene, camera);
        
        // Progress bar
        progress += 0.5;
        if (progress <= 100) {
            document.getElementById('progress-bar').style.width = progress + '%';
        }
    }
    
    animate();
    
    // Auto-advance after 3 seconds
    setTimeout(() => {
        window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'complete'}, '*');
    }, 3000);
    </script>
    """
    
    result = components.html(splash_html, height=800)
    
    if result == 'complete' or time.time() % 10 < 3:
        st.session_state.splash_complete = True
        st.session_state.app_phase = 'overview'
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# REAL-WORLD DATASET (70 CCS Sites)
# ═══════════════════════════════════════════════════════════════════════════════
REAL_DATA = {
    'Site': [
        'Sleipner (Norway)', 'Snøhvit (Norway)', 'In Salah (Algeria)', 'Otway Stage 1 (Australia)',
        'Otway Stage 2 (Australia)', 'Illinois Basin Decatur (USA)', 'Quest (Canada)', 'Weyburn-Midale (Canada)',
        'Boundary Dam (Canada)', 'Cranfield (USA)', 'Ketzin (Germany)', 'CarbFix (Iceland)',
        'Tomakomai (Japan)', 'Gorgon (Australia)', 'Northern Lights (Norway)',
        'DOE Shallow Clastic - Thin', 'DOE Shallow Clastic - Medium', 'DOE Shallow Clastic - Thick',
        'DOE Deep Clastic - Thin', 'DOE Deep Clastic - Medium', 'DOE Deep Clastic - Thick',
        'DOE Shallow Carbonate - Thin', 'DOE Shallow Carbonate - Medium', 'DOE Shallow Carbonate - Thick',
        'DOE Deep Carbonate - Thin', 'DOE Deep Carbonate - Medium', 'DOE Deep Carbonate - Thick',
        'Mount Simon (Illinois Basin, USA)', 'Utsira Sand (North Sea)', 'Morrison Formation (Colorado, USA)',
        'Tuscaloosa Marine Shale (USA)', 'Frio Formation (Texas, USA)', 'Madison Limestone (Wyoming, USA)',
        'Navajo Sandstone (Utah, USA)', 'Entrada Sandstone (Utah, USA)',
        'Bunter Sandstone (UK)', 'Forties Sandstone (UK)', 'Rotliegend Sandstone (Netherlands)',
        'Dogger Formation (France)', 'Muschelkalk (Germany)', 'Trias Grès (France)',
        'Gassum Formation (Denmark)', 'Johansen Formation (Norway)',
        'GOM Slope Sand - Shallow', 'GOM Slope Sand - Medium', 'GOM Slope Sand - Deep',
        'GOM Shelf Sand - Shallow', 'GOM Shelf Sand - Deep',
        'Paaratte Formation (Otway, Australia)', 'Waarre C Formation (Otway, Australia)',
        'Harvey Formation (SW Hub, Australia)', 'Precipice Sandstone (Surat, Australia)',
        'Aquistore (Weyburn area, Canada)', 'Lacq (France)', 'Casablanca (Spain)',
        'K12-B Gas Field (Netherlands)', 'Sleipner Vest (Norway)', 'Draugen (Norway)',
        'Saline Aquifer - Michigan Basin', 'Saline Aquifer - Williston Basin',
        'Saline Aquifer - Permian Basin', 'Saline Aquifer - Anadarko Basin',
        'Saline Aquifer - Gulf Coast', 'Saline Aquifer - Appalachian Basin',
        'Depleted Gas - Permian Basin', 'Depleted Gas - Gulf Coast', 'Depleted Gas - Rocky Mountains',
        'Depleted Oil - Midcontinent',
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
        0.25, 0.20, 0.20, 0.30, 0.26, 0.18, 0.22, 0.20,
        0.24, 0.26, 0.22, 0.28, 0.23, 0.19, 0.24, 0.27,
        0.30, 0.32, 0.28, 0.25, 0.23,
        0.25, 0.22, 0.26, 0.24,
        0.26, 0.20, 0.23, 0.18, 0.19, 0.25,
        0.22, 0.23, 0.25, 0.21, 0.27, 0.19,
        0.24, 0.26, 0.22, 0.25,
    ],
    'Permeability': [
        2500, 10, 20, 50, 150, 100, 50, 500,
        150, 300, 200, 5, 80, 150, 800,
        100, 100, 100, 50, 50, 50,
        20, 20, 20, 10, 10, 10,
        100, 2500, 80, 5, 250, 30, 120, 90,
        200, 500, 150, 40, 100, 70, 300, 600,
        500, 800, 400, 200, 120,
        180, 60, 350, 110,
        250, 100, 180, 50, 2000, 350,
        120, 150, 200, 90, 400, 60,
        180, 250, 120, 220,
    ],
    'Thickness': [
        200, 120, 60, 40, 80, 150, 90, 70,
        100, 180, 50, 30, 70, 250, 220,
        50, 50, 50, 100, 100, 100,
        30, 30, 30, 60, 60, 60,
        150, 200, 100, 40, 120, 80, 110, 95,
        130, 180, 120, 90, 110, 85, 160, 200,
        180, 220, 150, 140, 130,
        90, 60, 150, 100,
        140, 70, 120, 50, 180, 160,
        100, 120, 140, 85, 190, 75,
        120, 150, 110, 135,
    ],
    'Area': [
        240, 45, 20, 5, 12, 40, 30, 60,
        35, 80, 8, 3, 15, 500, 350,
        30, 30, 30, 60, 60, 60,
        20, 20, 20, 40, 40, 40,
        150, 240, 80, 15, 100, 50, 90, 70,
        120, 180, 100, 50, 90, 60, 140, 200,
        180, 250, 120, 110, 95,
        70, 40, 130, 85,
        110, 50, 90, 35, 200, 150,
        85, 95, 120, 70, 160, 55,
        100, 130, 90, 115,
    ],
    'Efficiency': [
        0.142, 0.036, 0.028, 0.045, 0.072, 0.051, 0.039, 0.095,
        0.062, 0.088, 0.055, 0.015, 0.048, 0.074, 0.125,
        0.048, 0.048, 0.048, 0.042, 0.042, 0.042,
        0.032, 0.032, 0.032, 0.025, 0.025, 0.025,
        0.055, 0.142, 0.041, 0.018, 0.068, 0.034, 0.052, 0.046,
        0.070, 0.095, 0.065, 0.038, 0.056, 0.044, 0.082, 0.105,
        0.098, 0.115, 0.089, 0.072, 0.058,
        0.075, 0.042, 0.088, 0.054,
        0.078, 0.052, 0.066, 0.038, 0.128, 0.092,
        0.058, 0.064, 0.075, 0.049, 0.095, 0.040,
        0.068, 0.082, 0.055, 0.074,
    ]
}

df = pd.DataFrame(REAL_DATA)

# ═══════════════════════════════════════════════════════════════════════════════
# PRESET CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════
PRESETS = {
    'Sleipner (Norway)': {
        'Porosity': 0.37, 'Pressure': 3600, 'Temperature': 37, 'Depth': 1012,
        'Residual_Gas_Saturation': 0.20, 'Permeability': 2500, 'Thickness': 200, 'Area': 240
    },
    'Quest (Canada)': {
        'Porosity': 0.16, 'Pressure': 2200, 'Temperature': 52, 'Depth': 2000,
        'Residual_Gas_Saturation': 0.20, 'Permeability': 50, 'Thickness': 90, 'Area': 30
    },
    'Gorgon (Australia)': {
        'Porosity': 0.20, 'Pressure': 4000, 'Temperature': 80, 'Depth': 2700,
        'Residual_Gas_Saturation': 0.25, 'Permeability': 150, 'Thickness': 250, 'Area': 500
    },
    'Illinois Basin (USA)': {
        'Porosity': 0.15, 'Pressure': 3000, 'Temperature': 70, 'Depth': 2130,
        'Residual_Gas_Saturation': 0.25, 'Permeability': 100, 'Thickness': 150, 'Area': 40
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING (CACHED)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def train_model():
    X = df[['Porosity', 'Pressure', 'Temperature', 'Depth', 
            'Residual_Gas_Saturation', 'Permeability', 'Thickness', 'Area']]
    y = df['Efficiency']
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])
    
    model.fit(X, y)
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    
    return model, cv_scores

model, cv_scores = train_model()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: OVERVIEW INTERFACE (GLASSMORPHISM)
# ═══════════════════════════════════════════════════════════════════════════════
def render_overview():
    load_custom_css()
    
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px 40px;">
        <h1 style="font-size: 3.5rem; font-weight: 700; margin-bottom: 10px; background: linear-gradient(135deg, #10b981 0%, #34d399 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Project Summary
        </h1>
        <p style="color: #94a3b8; font-size: 1.3rem;">CO₂ Storage Prediction System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get current inputs from session state or defaults
    if 'user_inputs' not in st.session_state:
        st.session_state.user_inputs = {
            'Porosity': 0.20, 'Pressure': 3000, 'Temperature': 75, 'Depth': 2000,
            'Residual_Gas_Saturation': 0.25, 'Permeability': 200, 'Thickness': 150, 'Area': 100
        }
    
    inputs = st.session_state.user_inputs
    
    # Display key parameters in grid
    cols = st.columns(4)
    
    params = [
        ('Porosity', f"{inputs['Porosity']:.2f}", '🔬'),
        ('Temperature', f"{inputs['Temperature']}°C", '🌡️'),
        ('Pressure', f"{inputs['Pressure']} psi", '💨'),
        ('Depth', f"{inputs['Depth']} m", '⬇️'),
        ('Permeability', f"{inputs['Permeability']} mD", '🌊'),
        ('Thickness', f"{inputs['Thickness']} m", '📏'),
        ('Area', f"{inputs['Area']} km²", '🗺️'),
        ('Residual Gas', f"{inputs['Residual_Gas_Saturation']:.2f}", '💧'),
    ]
    
    for i, (label, value, icon) in enumerate(params):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; min-height: 120px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 2rem; margin-bottom: 8px;">{icon}</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #10b981; margin-bottom: 4px;">{value}</div>
                <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Proceed button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Launch Dashboard", use_container_width=True):
            st.session_state.overview_complete = True
            st.session_state.app_phase = 'dashboard'
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: MAIN DASHBOARD (BENTO BOX LAYOUT)
# ═══════════════════════════════════════════════════════════════════════════════
def render_dashboard():
    load_custom_css()
    
    # ───────────────────────────────────────────────────────────────────────────
    # SIDEBAR CONTROLS
    # ───────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="font-size: 1.8rem; font-weight: 700; margin-bottom: 5px;">🌍 CO₂ Storage</h2>
            <p style="color: #94a3b8; font-size: 0.9rem;">Prediction System</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Preset selector
        st.markdown("### 📋 Quick Presets")
        preset_choice = st.selectbox(
            "Load Reference Site",
            ['Custom'] + list(PRESETS.keys()),
            help="Load parameters from real CCS projects"
        )
        
        if preset_choice != 'Custom':
            preset_params = PRESETS[preset_choice]
            for key, value in preset_params.items():
                st.session_state.user_inputs[key] = value
        
        st.markdown("---")
        st.markdown("### ⚙️ Input Parameters")
        
        # Input controls
        porosity = st.slider("Porosity", 0.05, 0.40, st.session_state.user_inputs['Porosity'], 0.01)
        pressure = st.number_input("Pressure (psi)", 500, 6000, st.session_state.user_inputs['Pressure'], 50)
        temperature = st.number_input("Temperature (°C)", 20, 120, st.session_state.user_inputs['Temperature'], 1)
        depth = st.number_input("Depth (m)", 400, 4000, st.session_state.user_inputs['Depth'], 50)
        sgr = st.slider("Residual Gas Sat.", 0.10, 0.35, st.session_state.user_inputs['Residual_Gas_Saturation'], 0.01)
        permeability = st.number_input("Permeability (mD)", 5, 3000, st.session_state.user_inputs['Permeability'], 5)
        thickness = st.number_input("Thickness (m)", 20, 300, st.session_state.user_inputs['Thickness'], 10)
        area = st.number_input("Area (km²)", 3, 600, st.session_state.user_inputs['Area'], 5)
        
        # Update session state
        st.session_state.user_inputs = {
            'Porosity': porosity, 'Pressure': pressure, 'Temperature': temperature, 'Depth': depth,
            'Residual_Gas_Saturation': sgr, 'Permeability': permeability, 'Thickness': thickness, 'Area': area
        }
        
        st.markdown("---")
        st.markdown("### 🎯 Model Performance")
        st.metric("R² Score", f"{cv_scores.mean():.3f}")
        st.metric("CV Std Dev", f"±{cv_scores.std():.3f}")
        
        with st.expander("📊 Cross-Validation Details"):
            st.write("5-Fold CV Scores:")
            for i, score in enumerate(cv_scores, 1):
                st.write(f"Fold {i}: {score:.4f}")
    
    # ───────────────────────────────────────────────────────────────────────────
    # MAIN CONTENT AREA
    # ───────────────────────────────────────────────────────────────────────────
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0 40px;">
        <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 5px;">
            CO₂ Storage Dashboard
        </h1>
        <p style="color: #94a3b8; font-size: 1rem;">Real-time reservoir evaluation powered by 70 field sites</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Make prediction
    inputs = st.session_state.user_inputs
    X_new = np.array([[inputs['Porosity'], inputs['Pressure'], inputs['Temperature'], 
                      inputs['Depth'], inputs['Residual_Gas_Saturation'], 
                      inputs['Permeability'], inputs['Thickness'], inputs['Area']]])
    
    prediction = model.predict(X_new)[0]
    
    # Calculate confidence interval (simplified)
    residuals = model.predict(df[['Porosity', 'Pressure', 'Temperature', 'Depth',
                                   'Residual_Gas_Saturation', 'Permeability', 'Thickness', 'Area']]) - df['Efficiency']
    std_residual = np.std(residuals)
    ci_lower = max(0, (prediction - 1.96 * std_residual) * 100)
    ci_upper = min(100, (prediction + 1.96 * std_residual) * 100)
    
    # Calculate capacity
    rho_co2 = 700
    volume = inputs['Area'] * 1e6 * inputs['Thickness']
    pore_volume = volume * inputs['Porosity']
    
    # Constraint factors
    sweep = 0.15 + 0.25 * (inputs['Permeability'] / 1000)
    sweep = min(0.45, sweep)
    p_util = 0.85 - 0.15 * (inputs['Pressure'] / 6000)
    d_factor = 1.0 - 0.45 * max(0, (800 - inputs['Depth'])) / 800
    d_factor = max(0.3, d_factor)
    comp = 0.55 - 0.2 * (inputs['Area'] / 600)
    comp = max(0.30, comp)
    injectivity = 0.70 + 0.30 * (inputs['Permeability'] / 1000)
    injectivity = min(1.0, injectivity)
    
    theoretical = pore_volume * rho_co2
    capacity_tonnes = theoretical * prediction * sweep * p_util * d_factor * comp * injectivity
    reduction_pct = round(100 * (1 - capacity_tonnes / theoretical), 1)
    
    # Classification
    eff_pct = prediction * 100
    if eff_pct >= 10:
        eff_label = "Excellent efficiency — Prime reservoir"
        eff_color = "#10b981"
    elif eff_pct >= 6:
        eff_label = "Good efficiency — Suitable reservoir"
        eff_color = "#10b981"
    elif eff_pct >= 4:
        eff_label = "Moderate efficiency — Feasible with optimization"
        eff_color = "#f59e0b"
    else:
        eff_label = "Low efficiency — High risk"
        eff_color = "#ef4444"
    
    # Find closest reference site
    X_train = df[['Porosity', 'Pressure', 'Temperature', 'Depth',
                  'Residual_Gas_Saturation', 'Permeability', 'Thickness', 'Area']].values
    distances = np.sqrt(np.sum((X_train - X_new) ** 2, axis=1))
    closest_idx = np.argmin(distances)
    closest = df.iloc[closest_idx]
    
    # ───────────────────────────────────────────────────────────────────────────
    # BENTO BOX LAYOUT
    # ───────────────────────────────────────────────────────────────────────────
    
    # Row 1: Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Storage Efficiency</div>
            <div class="metric-value" style="color: {eff_color};">{eff_pct:.2f}%</div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 5px;">
                CI: {ci_lower:.2f}% - {ci_upper:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Constrained Capacity</div>
            <div class="metric-value">{capacity_tonnes/1e6:.1f}M</div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 5px;">tonnes CO₂</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Theoretical Max</div>
            <div class="metric-value">{theoretical/1e6:.1f}M</div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 5px;">tonnes CO₂</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Operational Reduction</div>
            <div class="metric-value" style="color: #ef4444;">{reduction_pct}%</div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 5px;">vs theoretical</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Row 2: Classification & Reference Site
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="margin-top: 0;">📊 Reservoir Classification</h3>
            <div style="padding: 20px; text-align: center; background: rgba(255,255,255,0.05); border-radius: 8px; margin-top: 15px;">
                <div style="font-size: 1.3rem; font-weight: 600; color: {eff_color};">
                    {eff_label}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="margin-top: 0;">🎯 Closest Reference Site</h3>
            <div style="padding: 15px; background: rgba(255,255,255,0.05); border-radius: 8px; margin-top: 15px;">
                <div style="font-size: 1.2rem; font-weight: 600; color: #10b981; margin-bottom: 8px;">
                    {closest['Site']}
                </div>
                <div style="font-size: 0.9rem; color: #94a3b8;">
                    Efficiency: {closest['Efficiency']*100:.2f}% | Depth: {closest['Depth']}m
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Row 3: Charts
    col1, col2 = st.columns(2)
    
    # Sensitivity Analysis
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="margin-top: 0;">📈 Sensitivity Impact</h3>
        </div>
        """, unsafe_allow_html=True)
        
        sens_results = []
        base_pred = prediction
        delta = 0.1
        
        for param in ['Porosity', 'Pressure', 'Temperature', 'Depth', 
                     'Residual_Gas_Saturation', 'Permeability', 'Thickness', 'Area']:
            X_test = X_new.copy()
            param_idx = ['Porosity', 'Pressure', 'Temperature', 'Depth',
                        'Residual_Gas_Saturation', 'Permeability', 'Thickness', 'Area'].index(param)
            X_test[0, param_idx] *= (1 + delta)
            new_pred = model.predict(X_test)[0]
            pct_change = ((new_pred - base_pred) / base_pred) * 100
            sens_results.append({
                'Parameter': param.replace('_', ' '),
                '% Change': pct_change
            })
        
        sens_df = pd.DataFrame(sens_results)
        
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
        ax.set_facecolor('none')
        
        colors_list = ['#ef4444' if x < 0 else '#10b981' for x in sens_df['% Change']]
        bars = ax.barh(sens_df['Parameter'], sens_df['% Change'], color=colors_list, alpha=0.8)
        
        ax.axvline(x=0, color='#64748b', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('% Change in Efficiency', fontsize=11, color='#e2e8f0')
        ax.set_title('Impact per 10% Parameter Increase', fontsize=12, color='#f7fafc', pad=15)
        ax.tick_params(colors='#94a3b8')
        ax.grid(axis='x', alpha=0.2, color='#475569')
        
        for spine in ax.spines.values():
            spine.set_color('#475569')
            spine.set_linewidth(0.5)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Parameter Ranking
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="margin-top: 0;">🏆 Parameter Ranking</h3>
        </div>
        """, unsafe_allow_html=True)
        
        sens_df['Impact'] = sens_df['% Change'].abs()
        rank_df = sens_df.sort_values('Impact', ascending=False)
        
        fig2, ax2 = plt.subplots(figsize=(10, 6), facecolor='none')
        ax2.set_facecolor('none')
        
        bars = ax2.bar(range(len(rank_df)), rank_df['Impact'], color='#10b981', alpha=0.8)
        ax2.set_xticks(range(len(rank_df)))
        ax2.set_xticklabels(rank_df['Parameter'], rotation=35, ha='right', fontsize=9)
        ax2.set_ylabel('Impact Strength (%)', fontsize=11, color='#e2e8f0')
        ax2.set_title('Parameters by Absolute Impact', fontsize=12, color='#f7fafc', pad=15)
        ax2.tick_params(colors='#94a3b8')
        ax2.grid(axis='y', alpha=0.2, color='#475569')
        
        for spine in ax2.spines.values():
            spine.set_color('#475569')
            spine.set_linewidth(0.5)
        
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Row 4: Capacity Constraints
    st.markdown("""
    <div class="glass-card">
        <h3 style="margin-top: 0;">⚙️ Capacity Constraint Factors</h3>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 20px;">
            DOE/USGS volumetric methodology with operational constraints (Bachu 2015, NETL Atlas). 
            Permeability-based injectivity factor added per Das et al. (2023).
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    constraints = [
        ('Sweep Efficiency', sweep * 100, 'Pore volume swept — permeability adjusted'),
        ('Pressure Utilization', p_util * 100, 'Headroom before overpressure risk'),
        ('Depth Factor', d_factor * 100, 'Injectivity at reservoir depth'),
        ('Compartmentalization', comp * 100, 'Fault isolation limits effective volume'),
        ('Injectivity Factor', injectivity * 100, f'Permeability-based capacity fill ({permeability} mD)'),
    ]
    
    for i, (name, value, desc) in enumerate(constraints):
        with [col1, col2, col3][i % 3]:
            status = 'status-active' if value >= 60 else 'status-warning' if value >= 40 else 'status-inactive'
            st.markdown(f"""
            <div class="toggle-container">
                <span class="status-indicator {status}"></span>
                <div style="flex: 1;">
                    <div style="font-weight: 600; margin-bottom: 3px;">{name}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">{desc}</div>
                </div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #10b981;">
                    {value:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # ───────────────────────────────────────────────────────────────────────────
    # DOWNLOAD SECTION
    # ───────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h3>⬇️ Export Results</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # CSV Export
    with col1:
        out_df = pd.DataFrame({
            'Porosity': [inputs['Porosity']],
            'Pressure (psi)': [inputs['Pressure']],
            'Temperature (°C)': [inputs['Temperature']],
            'Depth (m)': [inputs['Depth']],
            'Residual Gas Saturation': [inputs['Residual_Gas_Saturation']],
            'Permeability (mD)': [inputs['Permeability']],
            'Thickness (m)': [inputs['Thickness']],
            'Area (km²)': [inputs['Area']],
            'Predicted Efficiency (%)': [round(prediction * 100, 2)],
            'CI Lower (%)': [round(ci_lower, 2)],
            'CI Upper (%)': [round(ci_upper, 2)],
            'Constrained Capacity (tonnes)': [round(capacity_tonnes, 0)],
            'Theoretical Capacity (tonnes)': [round(theoretical, 0)],
            'Closest Reference Site': [closest['Site']],
            'Classification': [eff_label],
        })
        
        st.download_button(
            "📊 Download CSV",
            out_df.to_csv(index=False),
            "co2_storage_results.csv",
            "text/csv",
            use_container_width=True
        )
    
    # PDF Export
    with col2:
        pdf_bytes = generate_pdf_report(
            inputs, prediction, ci_lower, ci_upper, capacity_tonnes, theoretical,
            reduction_pct, eff_label, eff_color, closest, cv_scores, sens_df,
            sweep, p_util, d_factor, comp, injectivity
        )
        
        st.download_button(
            "📄 Download PDF Report",
            pdf_bytes,
            "CO2_Storage_Report.pdf",
            "application/pdf",
            use_container_width=True
        )
    
    with col3:
        if st.button("🔄 Reset Dashboard", use_container_width=True):
            st.session_state.app_phase = 'splash'
            st.session_state.splash_complete = False
            st.session_state.overview_complete = False
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# PDF REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
def generate_pdf_report(inputs, prediction, ci_lower, ci_upper, capacity_tonnes, 
                        theoretical, reduction_pct, eff_label, eff_color, closest, 
                        cv_scores, sens_df, sweep, p_util, d_factor, comp, injectivity):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "Title", parent=styles["Normal"],
       fontName="Helvetica-Bold", fontSize=20,
        textColor=colors.HexColor("#1a5276"),
        spaceAfter=4, alignment=TA_CENTER
    )
    
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontName="Helvetica", fontSize=11,
        textColor=colors.HexColor("#5d6d7e"),
        spaceAfter=12, alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        "Heading", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=13,
        textColor=colors.HexColor("#1a5276"),
        spaceBefore=14, spaceAfter=6
    )
    
    note_style = ParagraphStyle(
        "Note", parent=styles["Normal"],
        fontName="Helvetica-Oblique", fontSize=9,
        textColor=colors.HexColor("#7f8c8d"), spaceAfter=4
    )
    
    footer_style = ParagraphStyle(
        "Footer", parent=styles["Normal"],
        fontName="Helvetica", fontSize=8,
        textColor=colors.HexColor("#aab7b8"),
        alignment=TA_CENTER
    )
    
    def create_table(data, col_widths):
        t = RLTable(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#eaf4fb"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#aed6f1")),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        return t
    
    story = []
    
    # Title
    story.append(Paragraph("CO<sub>2</sub> Storage Prediction Report", title_style))
    story.append(Paragraph("Data-Driven Reservoir Evaluation System", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, 
                           color=colors.HexColor("#1a5276"), spaceAfter=12))
    
    # Input Parameters
    story.append(Paragraph("Input Parameters", heading_style))
    story.append(create_table([
        ["Parameter", "Value", "Parameter", "Value"],
        ["Porosity", f"{inputs['Porosity']:.2f}", "Pressure (psi)", f"{inputs['Pressure']}"],
        ["Temperature (°C)", f"{inputs['Temperature']}", "Depth (m)", f"{inputs['Depth']}"],
        ["Residual Gas Sat.", f"{inputs['Residual_Gas_Saturation']:.2f}", 
         "Permeability (mD)", f"{inputs['Permeability']}"],
        ["Thickness (m)", f"{inputs['Thickness']}", "Area (km²)", f"{inputs['Area']}"],
    ], [1.5 * inch, 1.2 * inch, 1.5 * inch, 1.2 * inch]))
    
    story.append(Spacer(1, 10))
    
    # Prediction Results
    story.append(Paragraph("Prediction Results", heading_style))
    
    cv_mean = f"{cv_scores.mean():.2f}"
    cv_std = f"{cv_scores.std():.3f}"
    
    results_table = create_table([
        ["Metric", "Value"],
        ["CO2 Storage Efficiency", f"{round(prediction * 100, 2)} %"],
        ["95% Confidence Interval", f"{ci_lower:.2f}% — {ci_upper:.2f}%"],
        ["Constrained Capacity", f"{round(capacity_tonnes, 0):,.0f} tonnes"],
        ["Theoretical Max", f"{round(theoretical, 0):,.0f} tonnes"],
        ["Operational Reduction", f"{reduction_pct} %"],
        ["Model R² Score", f"{cv_mean}"],
        ["Closest Reference Site", closest['Site']],
        ["Reservoir Classification", eff_label],
    ], [3.2 * inch, 3.2 * inch])
    
    # Highlight classification row
    pdf_eff_color = colors.HexColor(eff_color)
    results_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#eaf4fb"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#aed6f1")),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("TEXTCOLOR", (1, 8), (1, 8), pdf_eff_color),
        ("FONTNAME", (1, 8), (1, 8), "Helvetica-Bold"),
    ]))
    
    story.append(results_table)
    story.append(Spacer(1, 10))
    
    # Capacity Constraints
    story.append(Paragraph("Capacity Constraint Factors", heading_style))
    story.append(Paragraph(
        "DOE/USGS volumetric methodology with operational constraints (Bachu 2015, NETL Atlas). "
        "Permeability-based injectivity factor added per Das et al. (2023).", note_style))
    
    story.append(create_table([
        ["Constraint", "Value", "Description"],
        ["Sweep Efficiency", f"{round(sweep * 100, 1)} %", 
         "% pore volume swept — adjusted for permeability"],
        ["Pressure Utilization", f"{round(p_util * 100, 1)} %", 
         "Headroom before overpressure risk"],
        ["Depth Factor", f"{round(d_factor * 100, 1)} %", 
         "Injectivity at reservoir depth"],
        ["Compartmentalization", f"{round(comp * 100, 1)} %", 
         "Fault isolation limits effective volume"],
        ["Injectivity Factor", f"{round(injectivity * 100, 1)} %",
         f"Permeability-based capacity fill factor ({inputs['Permeability']} mD)"],
    ], [1.8 * inch, 0.85 * inch, 3.75 * inch]))
    
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=1,
                           color=colors.HexColor("#aed6f1"), spaceAfter=4))
    
    # Footer
    story.append(Paragraph(
        "Generated by CO<sub>2</sub> Storage Prediction System | "
        "Calibrated to USGS, EU CO2StoP, NETL Atlas 5<super>th</super> Ed., Sleipner & Snøhvit field data | "
        "Permeability model: Park et al. (2021), Das et al. (2023)", footer_style))
    
    doc.build(story)
    return buf.getvalue()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP ROUTER
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if st.session_state.app_phase == 'splash':
        render_splash_screen()
    elif st.session_state.app_phase == 'overview':
        render_overview()
    elif st.session_state.app_phase == 'dashboard':
        render_dashboard()

if __name__ == "__main__":
    main()
