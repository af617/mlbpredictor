import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
import pickle
from io import StringIO

st.set_page_config(page_title="MLB Pitch Predictor v1.1.0", page_icon="⚾", layout="wide")

# Load model
@st.cache_resource
def load_model():
    with open("xgb_models.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Batter CSV
players_csv = """
player,height,OBP,k_pct,contact_pct,stance,bb_pct,ba
"Judge, Aaron",79,0.457,0.236,0.654,R,0.183,0.331
"Soto, Juan",74,0.396,0.192,0.776,L,0.178,0.263
"Stanton, Giancarlo",78,0.35,0.343,0.626,R,0.194,0.273
"Ohtani, Shohei",76,0.392,0.258,0.666,L,0.15,0.282
"""
players_df = pd.read_csv(StringIO(players_csv))

# Sidebar
st.sidebar.header("Pitch Inputs")
player = st.sidebar.selectbox("Select Batter", players_df['player'])
batter_stats = players_df[players_df['player'] == player].iloc[0]

pitch_type = st.sidebar.selectbox("Pitch Type", [
    "4-Seam Fastball", "Changeup", "Slider", "Sinker", "Cutter",
    "Split-Finger", "Curveball", "Knuckle Curve", "Slurve",
    "Sweeper", "Eephus", "Forkball", "Other", "Slow Curve"
])

release_speed = st.sidebar.slider("Release Speed (mph)", 70, 105, 95, 0.5)
effective_speed = st.sidebar.slider("Effective Speed (mph)", 70, 105, 95, 0.5)
release_spin_rate = st.sidebar.slider("Spin Rate (rpm)", 1500, 3500, 2300, 50)
plate_x = st.sidebar.slider("Horizontal Plate Location (ft)", -2.5, 2.5, 0.0, 0.01)
plate_z = st.sidebar.slider("Vertical Plate Location (ft)", 0.0, 5.0, 2.5, 0.01)
balls = st.sidebar.selectbox("Balls", [0,1,2,3])
strikes = st.sidebar.selectbox("Strikes", [0,1,2])
release_pos_y = st.sidebar.slider("Release Height (ft)", 3.0, 8.0, 6.0, 0.1)

# New sliders for batter swing
bat_speed = st.sidebar.slider("Bat Speed (mph)", 0, 120, 90, 1)
swing_length = st.sidebar.slider("Swing Length (ft)", 0.0, 6.0, 3.0, 0.01)
swung = 1 if bat_speed > 0 else 0

# Expected features
expected_features = [
    'release_speed', 'plate_x', 'plate_z', 'release_spin_rate', 'height', 'OBP',
    'k_pct', 'contact_pct', 'bb_pct', 'ba', 'balls', 'strikes', 'effective_speed',
    'p_throws', 'release_extension', 'release_pos_y', 'distance_from_center',
    'strikes_vs_balls', 'meatball', 'hittability', 'spin_effect',
    'stance_L','stance_R',
    'pitch_name_4-Seam Fastball','pitch_name_Changeup','pitch_name_Curveball','pitch_name_Cutter',
    'pitch_name_Eephus','pitch_name_Forkball','pitch_name_Knuckle Curve','pitch_name_Other',
    'pitch_name_Sinker','pitch_name_Slider','pitch_name_Slow Curve','pitch_name_Slurve',
    'pitch_name_Split-Finger','pitch_name_Sweeper',
    'count_0-0','count_0-1','count_0-2','count_1-0','count_1-1','count_1-2',
    'count_2-0','count_2-1','count_2-2','count_3-0','count_3-1','count_3-2',
    'bat_speed','swing_length','swung'
]

# Build input dict
input_dict = {feat: [0.0] for feat in expected_features}

# Derived features
distance_from_center = np.sqrt(plate_x**2 + (plate_z - 2.5)**2)
strikes_vs_balls = strikes / (balls+1)
meatball = 1.0 if (abs(plate_x) < 0.3 and abs(plate_z-2.5) < 0.3) else 0.0
hittability = batter_stats.contact_pct / (1+distance_from_center)
spin_effect = release_spin_rate * release_speed
p_throws = 0 # placeholder
release_extension = 6.0 # placeholder

# Fill features
input_dict.update({
    'release_speed':[release_speed],'plate_x':[plate_x],'plate_z':[plate_z],
    'release_spin_rate':[release_spin_rate],'effective_speed':[effective_speed],
    'release_pos_y':[release_pos_y],'balls':[balls],'strikes':[strikes],
    'height':[batter_stats.height],'OBP':[batter_stats.OBP],
    'k_pct':[batter_stats.k_pct],'contact_pct':[batter_stats.contact_pct],
    'bb_pct':[batter_stats.bb_pct],'ba':[batter_stats.ba],
    'distance_from_center':[distance_from_center],'strikes_vs_balls':[strikes_vs_balls],
    'meatball':[meatball],'hittability':[hittability],'spin_effect':[spin_effect],
    'p_throws':[p_throws],'release_extension':[release_extension],
    'bat_speed':[bat_speed],'swing_length':[swing_length],'swung':[swung]
})

# Pitch type
input_dict[f'pitch_name_{pitch_type}'] = [1.0]

# Stance
input_dict[f'stance_{batter_stats.stance}'] = [1.0]

# Count
input_dict[f'count_{balls}-{strikes}'] = [1.0]

# Convert to DataFrame
X_input = pd.DataFrame(input_dict)[expected_features]

# Prediction
if model:
    probs = model.predict_proba(X_input)[0]
    outcomes = ['Ball','Hit into Play','Strike']
    max_idx = np.argmax(probs)
    prediction = outcomes[max_idx]
    
    st.write(f"**Top Prediction:** {prediction} ({probs[max_idx]:.1%})")