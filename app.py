import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Car Price Predictor 🚗",
    page_icon="🚗",
    layout="centered"
)

# ==============================
# DARK THEME CUSTOM CSS
# ==============================
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: white;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD DATA (ONLINE - NO CSV ERROR)
# ==============================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/krishnaik06/Car-Price-Prediction/master/car%20data.csv"
    df = pd.read_csv(url)

    df = df.drop(['Car_Name'], axis=1)

    current_year = datetime.now().year
    df['car_age'] = current_year - df['Year']
    df = df.drop(['Year'], axis=1)

    df['Fuel_Type'] = df['Fuel_Type'].map({'Petrol':0, 'Diesel':1, 'CNG':2})
    df['Seller_Type'] = df['Seller_Type'].map({'Dealer':0, 'Individual':1})
    df['Transmission'] = df['Transmission'].map({'Manual':0, 'Automatic':1})

    return df

# ==============================
# TRAIN MODEL
# ==============================
@st.cache_resource
def train_model():
    df = load_data()
    X = df.drop(['Selling_Price'], axis=1)
    y = df['Selling_Price']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model

model = train_model()

# ==============================
# UI
# ==============================

st.title("🚗 Car Price Predictor")
st.markdown("### 💡 Get instant resale value of your car")

st.image("https://cdn.pixabay.com/photo/2012/05/29/00/43/car-49278_1280.jpg", use_container_width=True)

st.markdown("---")

# Layout
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("📅 Year", 2000, datetime.now().year, 2018)
    kms = st.number_input("🛣️ Kms Driven", 0, 200000, 30000)
    fuel = st.selectbox("⛽ Fuel Type", ["Petrol", "Diesel", "CNG"])

with col2:
    present_price = st.number_input("💰 Present Price (Lakhs)", 0.0, 50.0, 5.0)
    owner = st.selectbox("👤 Owner Count", [0,1,2,3])
    trans = st.selectbox("⚙️ Transmission", ["Manual", "Automatic"])

seller = st.selectbox("🏪 Seller Type", ["Dealer", "Individual"])

st.markdown("---")

# Mapping
fuel_map = {"Petrol":0, "Diesel":1, "CNG":2}
seller_map = {"Dealer":0, "Individual":1}
trans_map = {"Manual":0, "Automatic":1}

# Prediction
if st.button("🚀 Predict Price"):
    try:
        car_age = datetime.now().year - year

        data = np.array([[present_price, kms,
                          fuel_map[fuel],
                          seller_map[seller],
                          trans_map[trans],
                          owner,
                          car_age]])

        prediction = model.predict(data)

        st.success("✅ Prediction Completed!")
        st.metric("💰 Estimated Price", f"₹ {round(prediction[0], 2)} Lakhs")

        st.balloons()

    except Exception as e:
        st.error(f"❌ Error: {e}")