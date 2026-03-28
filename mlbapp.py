import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import pickle
from io import StringIO

# Page Config
st.set_page_config(page_title="MLB Pitch Predictor", page_icon="⚾", layout="wide")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .sidebar .sidebar-content { background-image: linear-gradient(#2e7bcf,#2e7bcf); color: white; }
    </style>
    """, unsafe_allow_stdio=True)

# -----------------------------

# -----------------------------
@st.cache_resource
def load_model():
    # Wrap in try-except for deployment safety
    try:
        with open("xgb_models.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file 'xgb_models.pkl' not found. Please upload it to your repo.")
        return None

model = load_model()

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
players_df = pd.read_csv(StringIO(players_csv))

# -----------------------------

# -----------------------------
st.sidebar.header("🕹️ Pitch Control Room")

with st.sidebar:
    player = st.selectbox("Select Batter", players_df['player'])
    batter_stats = players_df[players_df['player'] == player].iloc[0]
    
    st.divider()
    
    pitch_type = st.selectbox("Pitch Selection", 
        ["4-Seam Fastball", "Changeup", "Slider", "Sinker", "Cutter", "Split-Finger", "Curveball", "Knuckle Curve", "Slurve", "Sweeper"])
    
    st.subheader("Pitch Physics")
    release_speed = st.slider("Velocity (mph)", 70.0, 105.0, 95.0, 0.5)
    effective_speed = st.slider("Effective Velocity (mph)", 70.0, 105.0, 95.0, 0.5)
    release_spin_rate = st.slider("Spin Rate (rpm)", 1500.0, 3500.0, 2300.0, 50.0)
    
    st.subheader("Plate Location")
    plate_x = st.slider("Horizontal (Left/Right)", -2.5, 2.5, 0.0, 0.01)
    plate_z = st.slider("Vertical (Height)", 0.0, 5.0, 2.5, 0.01)
    
    st.subheader("Game Situation")
    b_col, s_col = st.columns(2)
    with b_col: balls = st.selectbox("Balls", [0,1,2,3])
    with s_col: strikes = st.selectbox("Strikes", [0,1,2])
    
    release_pos_y = st.slider("Release Height (ft)", 3.0, 8.0, 6.0, 0.1)

# -----------------------------

# -----------------------------
st.title("⚾ MLB Pitch Analysis Dashboard")
st.markdown(f"**Analyzing match-up:** Pitcher vs. **{player}**")

# Metrics Row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Batter Stance", batter_stats.stance)
m2.metric("Avg OBP", f"{batter_stats.OBP:.3f}")
m3.metric("Contact %", f"{batter_stats.contact_pct*100:.1f}%")
m4.metric("Current Count", f"{balls}-{strikes}")

st.divider()

# Logic for features (Simplified as per previous fix)
expected_features = [
    'release_speed', 'plate_x', 'plate_z', 'release_spin_rate', 'height', 'OBP', 
    'k_pct', 'contact_pct', 'bb_pct', 'ba', 'balls', 'strikes', 'effective_speed', 
    'p_throws', 'release_extension', 'release_pos_y', 'distance_from_center', 
    'strikes_vs_balls', 'meatball', 'hittability', 'spin_effect', 'stance_L', 
    'stance_R', 'pitch_name_4-Seam Fastball', 'pitch_name_Changeup', 
    'pitch_name_Curveball', 'pitch_name_Cutter', 'pitch_name_Eephus', 
    'pitch_name_Forkball', 'pitch_name_Knuckle Curve', 'pitch_name_Other', 
    'pitch_name_Sinker', 'pitch_name_Slider', 'pitch_name_Slow Curve', 
    'pitch_name_Slurve', 'pitch_name_Split-Finger', 'pitch_name_Sweeper', 
    'count_0-0', 'count_0-1', 'count_0-2', 'count_1-0', 'count_1-1', 
    'count_1-2', 'count_2-0', 'count_2-1', 'count_2-2', 'count_3-0', 
    'count_3-1', 'count_3-2'
]

input_dict = {feat: [0.0] for feat in expected_features}
input_dict.update({
    'release_speed': [float(release_speed)], 'plate_x': [float(plate_x)], 'plate_z': [float(plate_z)],
    'release_spin_rate': [float(release_spin_rate)], 'effective_speed': [float(effective_speed)],
    'release_pos_y': [float(release_pos_y)], 'balls': [float(balls)], 'strikes': [float(strikes)],
    'height': [float(batter_stats.height)], 'OBP': [float(batter_stats.OBP)],
    'k_pct': [float(batter_stats.k_pct)], 'contact_pct': [float(batter_stats.contact_pct)],
    'bb_pct': [float(batter_stats.bb_pct)], 'ba': [float(batter_stats.ba)],
    'distance_from_center': [np.sqrt(plate_x**2 + (plate_z - 2.5)**2)],
    'strikes_vs_balls': [strikes / (balls + 1)],
    'meatball': [1.0 if (abs(plate_x) < 0.3 and abs(plate_z - 2.5) < 0.3) else 0.0]
})
input_dict[f'pitch_name_{pitch_type}'] = [1.0]
input_dict[f'stance_{batter_stats.stance}'] = [1.0]
input_dict[f'count_{balls}-{strikes}'] = [1.0]

X_input = pd.DataFrame(input_dict)[expected_features]

# -----------------------------

# -----------------------------
viz_col, pred_col = st.columns([1, 1])

if model:
    probs = model.predict_proba(X_input)[0]
    outcomes = ['Ball', 'Strike', 'In Play']
    
    with viz_col:
        st.subheader("🎯 Pitch Location")
        fig, ax = plt.subplots(figsize=(5, 6))
        fig.patch.set_facecolor('#f5f7f9')
        
        # Strike Zone (approximate 17" wide, knee to chest)
        ax.add_patch(Rectangle((-0.85, 1.6), 1.7, 1.8, edgecolor='#333333', facecolor='#e1e8ef', alpha=0.5, linewidth=3, label="Strike Zone"))
        
        # Inner "Heart" of the zone
        ax.add_patch(Rectangle((-0.5, 2.0), 1.0, 1.0, edgecolor='#cc0000', facecolor='none', linestyle='--', alpha=0.3))
        
        # The Ball
        ax.add_patch(Circle((plate_x, plate_z), 0.12, color='#cc0000', zorder=5))
        ax.annotate("Pitch", (plate_x + 0.15, plate_z), fontsize=10, fontweight='bold')

        # Formatting
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0, 5.5)
        ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xticks([-2, -0.85, 0, 0.85, 2])
        ax.set_title(f"{pitch_type} vs {player}", fontsize=12)
        st.pyplot(fig)

    with pred_col:
        st.subheader("📊 Probability Breakdown")
        
        # Create a nice colored bar chart
        chart_data = pd.DataFrame({'Outcome': outcomes, 'Probability': probs})
        
        # Display as a table first for clarity
        st.dataframe(chart_data.set_index('Outcome').style.format("{:.1%}"), use_container_width=True)
        
        # Custom bar chart using Streamlit native
        st.bar_chart(chart_data.set_index('Outcome'))
        
        # Summary Text
        max_idx = np.argmax(probs)
        st.info(f"💡 **Prediction:** The most likely outcome is a **{outcomes[max_idx]}** ({probs[max_idx]:.1%})")