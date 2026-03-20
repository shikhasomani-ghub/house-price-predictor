import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="Real Estate AI", layout="wide")

# ======================
# DARK UI
# ======================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
.block-container {
    background-color: transparent;
}
h1, h2, h3 {
    color: #38bdf8;
}
p, label, span {
    color: white !important;
}
.card {
    background: rgba(255, 255, 255, 0.05);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    backdrop-filter: blur(10px);
}
.result-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}
.stButton>button {
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD MODEL
# ======================
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

# ======================
# HEADER
# ======================
st.markdown("""
<h1 style='text-align:center;'>🏙️ Real Estate AI</h1>
<p style='text-align:center;'>Smart property valuation powered by ML</p>
""", unsafe_allow_html=True)

st.divider()

# ======================
# INPUTS
# ======================
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🏠 Property Details")
    area = st.slider("Area (sq ft)", 500, 10000, 2000)
    bedrooms = st.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.slider("Bathrooms", 1, 5, 2)
    stories = st.slider("Stories", 1, 4, 2)
    parking = st.slider("Parking", 0, 5, 1)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("⚙️ Features")
    mainroad = st.toggle("Main Road")
    guestroom = st.toggle("Guest Room")
    basement = st.toggle("Basement")
    hotwaterheating = st.toggle("Hot Water Heating")
    airconditioning = st.toggle("Air Conditioning")
    prefarea = st.toggle("Preferred Area")

    furnishingstatus = st.selectbox(
        "Furnishing",
        ["furnished", "semi-furnished", "unfurnished"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ======================
# PREPROCESS
# ======================
def preprocess():
    data = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'parking': parking,
        'mainroad': int(mainroad),
        'guestroom': int(guestroom),
        'basement': int(basement),
        'hotwaterheating': int(hotwaterheating),
        'airconditioning': int(airconditioning),
        'prefarea': int(prefarea),
    }

    df = pd.DataFrame([data])

    df['furnishingstatus_semi-furnished'] = 1 if furnishingstatus == "semi-furnished" else 0
    df['furnishingstatus_unfurnished'] = 1 if furnishingstatus == "unfurnished" else 0

    df = df.reindex(columns=columns, fill_value=0)

    return df

# ======================
# CATEGORY
# ======================
def category(price):
    if price < 4000000:
        return "💸 Budget"
    elif price < 8000000:
        return "🏠 Mid-Range"
    else:
        return "🏡 Luxury"

# ======================
# BUTTON
# ======================
st.divider()

if st.button("🔍 Predict Price"):

    with st.spinner("Analyzing property..."):
        time.sleep(1.5)

        input_df = preprocess()
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

    # ======================
    # RESULT UI
    # ======================
    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    st.markdown("### Estimated Property Value")
    st.markdown(f"# ₹ {int(prediction):,}")

    st.markdown(f"### {category(prediction)} Property")

    # Insights
    st.markdown("#### 📊 Insights")
    st.write(f"• Larger area increases price significantly")
    st.write(f"• Amenities like AC & preferred area boost value")
    st.write(f"• Bathrooms & parking add premium")

    st.markdown('</div>', unsafe_allow_html=True)