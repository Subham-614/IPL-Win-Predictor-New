import base64
import streamlit as st
import pickle
import pandas as pd

# Cache background image data
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("background.jpg")

# Apply background styling
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"], [data-testid="stToolbar"] {{
background: rgba(0,0,0,0);
}}
</style>
"""

# Teams and venues
teams = ['--- select ---',
         'Sunrisers Hyderabad', 'Mumbai Indians', 'Kolkata Knight Riders',
         'Royal Challengers Bangalore', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']
cities = ['Bangalore', 'Hyderabad', 'Kolkata', 'Mumbai', 'Visakhapatnam',
          'Indore', 'Durban', 'Chandigarh', 'Delhi', 'Dharamsala', 'Ahmedabad',
          'Chennai', 'Ranchi', 'Nagpur', 'Mohali', 'Pune', 'Bengaluru', 'Jaipur',
          'Port Elizabeth', 'Centurion', 'Raipur', 'Sharjah', 'Cuttack',
          'Johannesburg', 'Cape Town', 'East London', 'Abu Dhabi', 'Kimberley', 'Bloemfontein']

pipe = pickle.load(open('pipe.pkl', 'rb'))

# Set page background
st.markdown(page_bg_img, unsafe_allow_html=True)

# Title
st.markdown("# **IPL VICTORY PREDICTOR**")

# Input fields
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select Batting Team', teams)
with col2:
    bowling_team = st.selectbox('Select Bowling Team',
                                 [team for team in teams if team != batting_team] if batting_team != '--- select ---' else teams)

selected_city = st.selectbox('Select Venue', cities)
target = st.number_input('Target Score', min_value=0)

col1, col2, col3 = st.columns(3)
with col1:
    score = st.number_input('Current Score', min_value=0)
with col2:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
with col3:
    wickets = st.number_input('Wickets Fallen', min_value=0, max_value=10)

if st.button('Predict Winning Probability'):
    try:
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_remaining = 10 - wickets
        crr = score / overs if overs > 0 else 0
        rrr = runs_left / (balls_left / 6) if balls_left > 0 else 0

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets_remaining': [wickets_remaining],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        result = pipe.predict_proba(input_df)
        win = round(result[0][1] * 100)
        loss = round(result[0][0] * 100)

        st.header(f"{batting_team} Win Probability: {win}%")
        st.header(f"{bowling_team} Win Probability: {loss}%")

    except Exception as e:
        st.error(f"Error: {str(e)}. Please check your inputs.")
