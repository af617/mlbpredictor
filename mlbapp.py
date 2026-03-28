import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
import pickle
from io import StringIO

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="MLB Pitch Predictor v2.0", page_icon="⚾", layout="wide")
HIT_THRESHOLD = 0.30

st.markdown("""
<style>
.main { background-color: #f5f7f9; }
.stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    with open("xgb_models.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []


# =============================
# BATTER CSV
# =============================
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
"Rodriguez, Julio",76,0.324,0.214,0.718,R,0.062,0.267
"""
players_df = pd.read_csv(StringIO(players_csv))


# =============================
# SIDEBAR INPUTS
# =============================
st.sidebar.header("⚾ Pitch Lab")

player = st.sidebar.selectbox("Select Batter", players_df["player"])
batter_stats = players_df[players_df["player"] == player].iloc[0]

pitch_type = st.sidebar.selectbox(
    "Pitch Type",
    ["4-Seam Fastball", "Changeup", "Slider", "Sinker", "Cutter",
     "Split-Finger", "Curveball", "Knuckle Curve", "Slurve", "Sweeper"]
)

release_speed = st.sidebar.slider("Release Speed", 70.0, 105.0, 95.0, 0.5)
effective_speed = st.sidebar.slider("Effective Speed", 70.0, 105.0, 95.0, 0.5)
release_spin_rate = st.sidebar.slider("Spin Rate", 1500.0, 3500.0, 2300.0, 50.0)
plate_x = st.sidebar.slider("Plate X", -2.5, 2.5, 0.0, 0.01)
plate_z = st.sidebar.slider("Plate Z", 0.0, 5.0, 2.5, 0.01)
balls = st.sidebar.selectbox("Balls", [0, 1, 2, 3])
strikes = st.sidebar.selectbox("Strikes", [0, 1, 2])
release_pos_y = st.sidebar.slider("Release Pos Y", 45.0, 60.0, 54.0, 0.1)
release_extension = st.sidebar.slider("Release Extension", 5.0, 8.0, 6.5, 0.1)

# NEW TOP FEATURES
swing = st.sidebar.selectbox("Did Batter Swing?", [0, 1])
bat_speed = st.sidebar.slider("Bat Speed", 40.0, 90.0, 70.0, 0.5)
swing_length = st.sidebar.slider("Swing Length", 5.0, 25.0, 12.0, 0.5)


# =============================
# FEATURE ENGINEERING
# =============================
count = f"{balls}-{strikes}"
distance_from_center = np.sqrt(plate_x**2 + (plate_z - 2.5)**2)
strikes_vs_balls = int(strikes > balls)
meatball = int(
    pitch_type in ["4-Seam Fastball", "Changeup"]
    and abs(plate_x) < 0.5
    and 1.5 <= plate_z <= 3.5
)
hittability = batter_stats.contact_pct / (1 + distance_from_center)
spin_effect = release_spin_rate * release_speed


# =============================
# BUILD INPUT ROW
# =============================
X_input = pd.DataFrame([{
    "release_speed": release_speed,
    "plate_x": plate_x,
    "plate_z": plate_z,
    "release_spin_rate": release_spin_rate,
    "height": batter_stats.height,
    "OBP": batter_stats.OBP,
    "k_pct": batter_stats.k_pct,
    "contact_pct": batter_stats.contact_pct,
    "bb_pct": batter_stats.bb_pct,
    "ba": batter_stats.ba,
    "balls": balls,
    "strikes": strikes,
    "effective_speed": effective_speed,
    "p_throws": 0,
    "release_extension": release_extension,
    "release_pos_y": release_pos_y,
    "distance_from_center": distance_from_center,
    "strikes_vs_balls": strikes_vs_balls,
    "meatball": meatball,
    "hittability": hittability,
    "spin_effect": spin_effect,
    "swing": swing,
    "bat_speed": bat_speed,
    "swing_length": swing_length,
    f"stance_{batter_stats.stance}": 1,
    f"pitch_name_{pitch_type}": 1,
    f"count_{count}": 1,
}]).fillna(0)


# align with trained columns
if hasattr(model, "feature_names_in_"):
    X_input = X_input.reindex(columns=model.feature_names_in_, fill_value=0)


# =============================
# PREDICTION + THRESHOLD
# =============================
probs = model.predict_proba(X_input)[0]

# class order from your latest model:
# 0 = ball, 1 = hit_into_play, 2 = strike
if probs[1] >= HIT_THRESHOLD:
    pred_idx = 1
else:
    pred_idx = 0 if probs[0] >= probs[2] else 2

outcomes = ["Ball", "In Play", "Strike"]
prediction = outcomes[pred_idx]


# =============================
# UI OUTPUT
# =============================
st.title("⚾ MLB Pitch Predictor")
st.subheader(f"{player} vs {pitch_type}")

left, right = st.columns(2)

with left:
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.add_patch(Rectangle((-0.85, 1.6), 1.7, 1.8, fill=False, linewidth=2))

    batter_x = 1.6 if batter_stats.stance == "R" else -1.6
    ax.add_patch(Rectangle((batter_x - 0.2, 1.5), 0.4, 2.8, alpha=0.5))
    ax.add_patch(Circle((batter_x, 4.5), 0.2, alpha=0.5))

    ax.add_patch(Circle((plate_x, plate_z), 0.12))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0, 5.5)
    ax.grid(alpha=0.3)
    st.pyplot(fig)

with right:
    st.subheader("📊 Outcome Probabilities")
    prob_df = pd.DataFrame({
        "Outcome": outcomes,
        "Probability": [probs[0], probs[1], probs[2]]
    })
    st.bar_chart(prob_df.set_index("Outcome"))
    st.success(f"Prediction: {prediction}")
    st.write(f"Threshold used for In Play: {HIT_THRESHOLD}")


# =============================
# HISTORY
# =============================
st.session_state.history.append({
    "Batter": player,
    "Pitch": pitch_type,
    "Prediction": prediction,
    "In Play Prob": round(float(probs[1]), 3)
})

st.subheader("📜 Recent Predictions")
st.dataframe(pd.DataFrame(st.session_state.history).tail(10), use_container_width=True)
