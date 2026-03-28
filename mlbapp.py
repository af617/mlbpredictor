st.sidebar.header("🕹️ Pitch Control Room")

with st.sidebar:
    player = st.selectbox("Select Batter", players_df['player'])
    batter_stats = players_df[players_df['player'] == player].iloc[0]
    st.divider()
    
    pitch_type = st.selectbox(
        "Pitch Selection", 
        ["4-Seam Fastball", "Changeup", "Slider", "Sinker", "Cutter", "Split-Finger", 
         "Curveball", "Knuckle Curve", "Slurve", "Sweeper"]
    )
    
    release_speed = st.slider("Velocity (mph)", 70.0, 105.0, 95.0, 0.5, help="Pitch speed in miles per hour")
    effective_speed = st.slider("Effective Velocity (mph)", 70.0, 105.0, 95.0, 0.5, help="Adjusted pitch speed factoring in spin and movement")
    release_spin_rate = st.slider("Spin Rate (rpm)", 1500.0, 3500.0, 2300.0, 50.0, help="Spin of the pitch in revolutions per minute")
    
    plate_x = st.slider("Horizontal Position (ft, Plate X)", -2.5, 2.5, 0.0, 0.01)
    plate_z = st.slider("Vertical Position (ft, Plate Z)", 0.0, 5.0, 2.5, 0.01)
    
    bat_speed = st.slider("Bat Speed (mph)", 0.0, 120.0, 80.0, 0.5, help="Bat speed in miles per hour")
    swing_length = st.slider("Swing Length (ft)", 0.0, 10.0, 5.0, 0.1, help="Estimated swing path length in feet")
    swing = 0 if bat_speed == 0 else 1  # automatic calculation of swung
    
    release_pos_y = st.slider("Release Height (ft)", 3.0, 8.0, 6.0, 0.1)
    
    b_col, s_col = st.columns(2)
    with b_col:
        balls = st.selectbox("Balls", [0,1,2,3])
    with s_col:
        strikes = st.selectbox("Strikes", [0,1,2])
    
    st.divider()
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()