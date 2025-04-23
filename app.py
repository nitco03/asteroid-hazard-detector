import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go

# Load trained model
model = joblib.load('xgboost_asteroid_model.pkl')

# Custom threshold (for sensitivity)
THRESHOLD = 0.3

# Average values for visualization
hazardous_avg = {
    "Diameter Min": 0.5,
    "Diameter Max": 1.2,
    "Velocity": 60000,
    "Miss Distance": 40000000,
    "Magnitude": 19.5
}

non_hazardous_avg = {
    "Diameter Min": 0.3,
    "Diameter Max": 0.7,
    "Velocity": 30000,
    "Miss Distance": 60000000,
    "Magnitude": 21.5
}

def simulate_asteroid_position(miss_distance):
    au_distance = miss_distance / 150_000_000
    angle = np.random.uniform(0, 2 * np.pi)
    x = au_distance * np.cos(angle)
    y = au_distance * np.sin(angle)
    z = np.random.uniform(-0.1, 0.1)
    return x, y, z

def plot_solar_system_with_asteroid(miss_distance):
    x, y, z = simulate_asteroid_position(miss_distance)
    fig = go.Figure()

    # Sun
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0],
        mode='markers+text', marker=dict(size=10, color='yellow'),
        text=["☀ Sun"], name="Sun"
    ))

    # Earth's orbit
    orbit_x = np.cos(np.linspace(0, 2 * np.pi, 100))
    orbit_y = np.sin(np.linspace(0, 2 * np.pi, 100))
    orbit_z = np.zeros(100)
    fig.add_trace(go.Scatter3d(x=orbit_x, y=orbit_y, z=orbit_z,
        mode='lines', line=dict(color='blue', width=1), name='Earth Orbit'
    ))

    # Earth
    fig.add_trace(go.Scatter3d(x=[1], y=[0], z=[0],
        mode='markers+text', marker=dict(size=6, color='blue'),
        text=["🌍 Earth"], name="Earth"
    ))

    # Asteroid
    fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z],
        mode='markers+text', marker=dict(size=6, color='red'),
        text=["☄️ Asteroid"], name="Asteroid"
    ))

    fig.update_layout(
        title="🌌 3D Solar System Asteroid View (Simplified)",
        scene=dict(
            xaxis_title='X (AU)', yaxis_title='Y (AU)', zaxis_title='Z',
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig

# Streamlit UI
st.set_page_config(page_title="Asteroid Hazard Detector", layout="wide")
st.title("☄️ Asteroid Hazard Detection System")
st.markdown("Enter the asteroid's characteristics to determine if it's **potentially hazardous**.")

# Input section
col1, col2 = st.columns(2)

with col1:
    dmin = st.number_input("Estimated Diameter Min (km)", 0.0, 5.0, 0.5)
    velocity = st.number_input("Relative Velocity (km/h)", 0.0, 100000.0, 50000.0)
    magnitude = st.number_input("Absolute Magnitude", 0.0, 30.0, 20.0)

with col2:
    dmax = st.number_input("Estimated Diameter Max (km)", 0.0, 5.0, 1.0)
    distance = st.number_input("Miss Distance from Earth (km)", 0.0, 100_000_000.0, 50_000_000.0)

# Prediction button
if st.button("Predict Hazard"):
    input_data = np.array([[dmin, dmax, velocity, distance, magnitude]])
    proba = model.predict_proba(input_data)[0][1]
    pred = 1 if proba >= THRESHOLD else 0

    st.markdown(f"### 🧪 Hazard Probability: `{proba:.4f}`")

    if pred == 1:
        st.markdown("## 🚨 Result: **Hazardous Asteroid Detected!**", unsafe_allow_html=True)
        st.error("⚠️ This asteroid might be potentially hazardous based on its characteristics.")
    else:
        st.markdown("## ✅ Result: **This asteroid is not hazardous**", unsafe_allow_html=True)
        st.success("✔️ This asteroid is not likely to be hazardous.")

    # 3D Visualization
    st.markdown("### 🌠 Asteroid Position in the Solar System")
    fig = plot_solar_system_with_asteroid(distance)
    st.plotly_chart(fig, use_container_width=True)

    # Comparison Chart
    st.markdown("### 📊 Asteroid Metrics Comparison")
    user_input = {
        "Diameter Min": dmin,
        "Diameter Max": dmax,
        "Velocity": velocity,
        "Miss Distance": distance,
        "Magnitude": magnitude
    }

    comparison_df = pd.DataFrame(
        [user_input, hazardous_avg, non_hazardous_avg],
        index=["This Asteroid", "Hazardous Avg", "Non-Hazardous Avg"]
    )

    st.bar_chart(comparison_df.T)
