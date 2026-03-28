import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import pickle  # Assuming you saved your trained XGBoost model

# -----------------------------
# 1️⃣ Load Model & Player Data
# -----------------------------
# Load trained model
with open("xgb_models.pkl", "rb") as f:
    model = pickle.load(f)

# Load batter stats
players_csv = """
player,height,OBP,k_pct,contact_pct,stance,bb_pct,ba
"Judge, Aaron",79,0.457,0.236,0.654,R,0.183,0.331
"Soto, Juan",74,0.396,0.192,0.776,L,0.178,0.263
"Stanton, Giancarlo",78,0.35,0.343,0.626,R,0.194,0.273
"Volpe, Anthony",70,0.272,0.252,0.749,R,0.072,0.212
"Ohtani, Shohei",76,0.392,0.258,0.666,L,0.15,0.282
"Betts, Mookie",69,0.326,0.103,0.847,R,0.092,0.258
"Alvarez, Yordan",77,0.367,0.166,0.776,L,0.141,0.273
"Acuna Jr., Ronald",72,0.417,0.248,0.697,R,0.172,0.29
"Tatis Jr., Fernando",75,0.368,0.187,0.73,R,0.129,0.268
"Carroll, Corbin",70,0.343,0.238,0.752,L,0.104,0.259
"Adell, Joey",74,0.303,0.265,0.715,R,0.058,0.236
"De La Cruz, Elly",78,0.336,0.259,0.683,L,0.096,0.264
"Freeman, Freddie",77,0.367,0.204,0.727,L,0.096,0.295
"Guerrero Jr., Vladimir",72,0.381,0.138,0.785,R,0.119,0.292
"Henderson, Gunnar",75,0.349,0.21,0.749,L,0.095,0.274
"Rodriguez, Julio",76,0.324,0.214,0.718,R,0.062,0.267
"Witt Jr., Bobby",73,0.351,0.182,0.761,R,0.071,0.295
"Riley, Austin",74,0.309,0.286,0.705,R,0.06,0.26
"Seager, Corey",75,0.373,0.196,0.721,L,0.13,0.271
"Semien, Marcus",72,0.305,0.174,0.774,R,0.094,0.23
"Tucker, Kyle",75,0.377,0.148,0.798,L,0.146,0.266
"Buxton, Byron",73,0.327,0.273,0.687,R,0.076,0.264
"""
from io import StringIO
players_df = pd.read_csv(StringIO(players_csv))

# -----------------------------
# 2️⃣ Streamlit Sidebar Inputs
# -----------------------------
st.title("⚾ Baseball Pitch Outcome Predictor")

# Player selection
player = st.selectbox("Select Batter", players_df['player'])
batter_stats = players_df[players_df['player'] == player].iloc[0]

# Pitch type
pitch_type = st.selectbox(
    "Pitch Type",
    ["4-Seam Fastball", "Changeup", "Slider", "Sinker", "Cutter", "Split-Finger", "Curveball", "Knuckle Curve", "Slurve", "Sweeper"]
)

# Sliders for pitch info
release_speed = st.slider("Release Speed (mph)", 70, 105, 95)
release_spin_rate = st.slider("Spin Rate (rpm)", 1800, 3000, 2300)
plate_x = st.slider("Plate X (feet, left=-, right=+)", -2.0, 2.0, 0, 0.01)
plate_z = st.slider("Plate Z (feet, bottom=0, top=5)", 0, 5.0, 2.5, 0.01)
balls = st.slider("Balls Count", 0, 3, 0)
strikes = st.slider("Strikes Count", 0, 2, 0)
effective_speed = st.slider("Effective Speed (mph)\nAdjusted for spin and movement", 70, 105, 95)
release_pos_y = st.slider("Release Pos Y (feet above ground)", 3, 8, 6)

# -----------------------------
# 3️⃣ Prepare Features for Model
# -----------------------------
# Combine player stats with pitch info
X_input = pd.DataFrame({
    'release_speed':[release_speed],
    'plate_x':[plate_x],
    'plate_z':[plate_z],
    'release_spin_rate':[release_spin_rate],
    'effective_speed':[effective_speed],
    'release_pos_y':[release_pos_y],
    'height':[batter_stats.height],
    'OBP':[batter_stats.OBP],
    'k_pct':[batter_stats.k_pct],
    'contact_pct':[batter_stats.contact_pct],
    'bb_pct':[batter_stats.bb_pct],
    'ba':[batter_stats.ba],
})

# One-hot encode pitch type
for pt in ["4-Seam Fastball", "Changeup", "Slider", "Sinker", "Cutter", "Split-Finger", "Curveball", "Knuckle Curve", "Slurve", "Sweeper"]:
    X_input[f'pitch_name_{pt}'] = 1 if pitch_type == pt else 0

# -----------------------------
# 4️⃣ Predict
# -----------------------------
probs = model.predict_proba(X_input)[0]
outcomes = ['ball', 'strike', 'hit_into_play']

# -----------------------------
# 5️⃣ Display Strikezone & Ball
# -----------------------------
fig, ax = plt.subplots(figsize=(4,6))
# Draw strikezone
ax.add_patch(Rectangle((-0.85,1.6), 1.7, 1.8, edgecolor='black', facecolor='none', linewidth=2))
# Draw ball
ax.add_patch(Circle((plate_x, plate_z), 0.1, color='red'))
ax.set_xlim(-2,2)
ax.set_ylim(0,5)
ax.set_xlabel("Plate X")
ax.set_ylabel("Plate Z")
ax.set_title("Strikezone")
st.pyplot(fig)

# -----------------------------
# 6️⃣ Display Probability Chart
# -----------------------------
st.subheader("Predicted Probabilities")
prob_df = pd.DataFrame({'Outcome': outcomes, 'Probability': probs})
st.bar_chart(prob_df.set_index('Outcome'))