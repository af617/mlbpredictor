import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import pickle
from io import StringIO
import altair as alt
import plotly.graph_objects as go

st.set_page_config(page_title="MLB Pitch Predictor v1.2.0", page_icon="⚾", layout="wide")
HIT_THRESHOLD = 0.30

st.markdown("""
<style>
.main { background-color: #f5f7f9; }
.stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open("batspeeds_nopfx_inplay.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

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

st.sidebar.header("⚾ Pitch Lab")

player = st.sidebar.selectbox(
    "Select Batter", 
    players_df["player"],
    help="The specific MLB player currently at the plate."
)
batter_stats = players_df[players_df["player"] == player].iloc[0]

pitch_type = st.sidebar.selectbox(
    "Pitch Type",
    ["4-Seam Fastball", "Changeup", "Slider", "Sinker", "Cutter",
     "Split-Finger", "Curveball", "Knuckle Curve", "Slurve", "Sweeper"],
    help="The classification of the pitch based on its grip and movement profile."
)

release_speed = st.sidebar.slider(
    "Velocity: Release Speed (mph)", 70.0, 105.0, 95.0, 0.5,
    help="The maximum velocity of the ball at the moment it leaves the pitcher's hand."
)
effective_speed = st.sidebar.slider(
    "Velocity: Effective Speed (mph)", 70.0, 105.0, 95.0, 0.5,
    help="The velocity of the pitch adjusted for perceived speed based on the pitcher's extension."
)
release_spin_rate = st.sidebar.slider(
    "Movement: Spin Rate (rpm)", 1500.0, 3500.0, 2300.0, 50.0,
    help="How fast the ball is spinning in revolutions per minute. Higher spin often creates more movement."
)
plate_x = st.sidebar.slider(
    "Location: Horizontal Plate X (ft)", -2.5, 2.5, 0.0, 0.01,
    help="The horizontal position of the ball as it crosses the plate (0.0 is dead center)."
)
plate_z = st.sidebar.slider(
    "Location: Vertical Plate Z (ft)", 0.0, 5.0, 2.5, 0.01,
    help="The height of the ball as it crosses the plate in feet."
)
balls = st.sidebar.selectbox(
    "Situation: Balls", [0, 1, 2, 3],
    help="Current number of balls in the count."
)
strikes = st.sidebar.selectbox(
    "Situation: Strikes", [0, 1, 2],
    help="Current number of strikes in the count."
)
release_pos_y = st.sidebar.slider(
    "Release: Y-Position (ft)", 45.0, 60.0, 54.0, 0.1,
    help="The distance from the back of home plate where the ball is released."
)
release_extension = st.sidebar.slider(
    "Release: Extension (ft)", 5.0, 8.0, 6.5, 0.1,
    help="How far the pitcher releases the ball in front of the rubber."
)

swing_choice = st.sidebar.radio(
    "Swing?", ["Yes", "No"], index=1,
    help="Indicates whether the batter attempts to swing at the pitch."
)
swing = 1 if swing_choice == "Yes" else 0

if swing == 1:
    bat_speed = st.sidebar.slider(
        "Swing: Bat Speed (mph)", 40.0, 90.0, 70.0, 0.5,
        help="The speed of the sweet spot of the bat at the moment of impact."
    )
    swing_length = st.sidebar.slider(
        "Swing: Swing Length (ft)", 5.0, 25.0, 12.0, 0.5,
        help="The total distance the bat head travels through the swing path."
    )
else:
    bat_speed = 0.0
    swing_length = 0.0

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

if hasattr(model, "feature_names_in_"):
    X_input = X_input.reindex(columns=model.feature_names_in_, fill_value=0)

probs = model.predict_proba(X_input)[0]

if swing == 1:
    probs[0] = 0
    sum_remaining = probs[1] + probs[2]
    if sum_remaining > 0:
        probs[1] = probs[1] / sum_remaining
        probs[2] = probs[2] / sum_remaining
    else:
        probs[1], probs[2] = 0.5, 0.5
else:
    probs[1] = 0
    sum_remaining = probs[0] + probs[2]
    if sum_remaining > 0:
        probs[0] = probs[0] / sum_remaining
        probs[2] = probs[2] / sum_remaining
    else:
        probs[0], probs[2] = 0.5, 0.5

if probs[1] >= HIT_THRESHOLD:
    pred_idx = 1
else:
    pred_idx = 0 if probs[0] >= probs[2] else 2

outcomes = ["Ball", "In Play", "Strike"]
prediction = outcomes[pred_idx]

st.title("⚾ MLB Pitch Predictor v1.1.2")
st.subheader(f"{player} vs {pitch_type}")

if swing == 1:
    st.info(f"Swing Detected: {bat_speed} mph")
else:
    st.warning("Batter Took Pitch: Ball/Strike logic active")

left, right = st.columns(2)

with left:
    fig_2d, ax_2d = plt.subplots(figsize=(5, 5))
    ax_2d.add_patch(Rectangle((-0.85, 1.6), 1.7, 1.8, fill=False, linewidth=2, color="black"))
    batter_x = 1.6 if batter_stats.stance == "R" else -1.6
    ax_2d.add_patch(Rectangle((batter_x - 0.2, 1.5), 0.4, 2.8, color="gray", alpha=0.3))
    ax_2d.add_patch(Circle((batter_x, 4.5), 0.2, color="gray", alpha=0.3))
    ax_2d.scatter(plate_x, plate_z, s=150, c="red", edgecolors="white", zorder=5)
    ax_2d.set_xlim(-2.5, 2.5)
    ax_2d.set_ylim(0, 5.5)
    ax_2d.set_title("2D Strike Zone (Umpire View)")
    ax_2d.grid(alpha=0.2)
    st.pyplot(fig_2d)

    y_traj = np.linspace(release_pos_y, 0, 50)
    x_traj = np.linspace(0, plate_x, 50)
    z_traj = np.linspace(6, plate_z, 50)

    fig_3d = go.Figure()
    hp_x = [-0.71, 0.71, 0.71, 0, -0.71, -0.71]
    hp_y = [0, 0, 0.5, 1.0, 0.5, 0]
    hp_z = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    fig_3d.add_trace(go.Scatter3d(x=hp_x, y=hp_y, z=hp_z, mode='lines', line=dict(color='white', width=6), name='Home Plate'))
    sz_x = [-0.85, 0.85, 0.85, -0.85, -0.85]; sz_z = [1.6, 1.6, 3.4, 3.4, 1.6]; sz_y = [0, 0, 0, 0, 0]
    fig_3d.add_trace(go.Scatter3d(x=sz_x, y=sz_y, z=sz_z, mode='lines', line=dict(color='rgba(0,0,0,0.5)', width=4), name='Strike Zone'))
    fig_3d.add_trace(go.Scatter3d(x=x_traj, y=y_traj, z=z_traj, mode='lines', line=dict(color='red', width=6), name='Pitch Path'))
    fig_3d.add_trace(go.Scatter3d(x=[plate_x], y=[0], z=[plate_z], mode='markers', marker=dict(size=10, color='white', line=dict(color='black', width=2)), name='Impact'))

    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(title='Width', range=[-4, 4]),
            yaxis=dict(title='Distance', range=[-5, 65]),
            zaxis=dict(title='Height', range=[0, 8]),
            camera=dict(eye=dict(x=0, y=1.8, z=0.8), center=dict(x=0, y=0, z=0.3), up=dict(x=0, y=0, z=1)),
            aspectmode='manual',
            aspectratio=dict(x=1, y=2.5, z=1)
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600
    )
    st.plotly_chart(fig_3d, use_container_width=True)

with right:
    st.subheader("📊 Outcome Probabilities")
    prob_df = pd.DataFrame({"Outcome": outcomes, "Probability": [probs[0], probs[1], probs[2]]})
    prob_chart = (alt.Chart(prob_df).mark_bar().encode(x=alt.X("Outcome", sort=None), y=alt.Y("Probability", scale=alt.Scale(domain=[0, 1])), color=alt.value("#1f77b4")).properties(height=350))
    st.altair_chart(prob_chart, use_container_width=True)
    color = "green" if prediction == "In Play" else "blue"
    st.markdown(f"### Prediction: :{color}[{prediction}]")
    st.write(f"Confidence Score: **{probs[pred_idx]:.1%}**")

st.session_state.history.append({"Batter": player, "Pitch": pitch_type, "Velo": release_speed, "Prediction": prediction, "In Play Prob": round(float(probs[1]), 3)})
st.subheader("Recent Predictions")
st.dataframe(pd.DataFrame(st.session_state.history).tail(10), use_container_width=True)

