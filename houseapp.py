import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

st.set_page_config(page_title="House Price Prediction", page_icon="🏡")


# Load California Housing Dataset
@st.cache_data
def load_data():
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["PRICE"] = housing.target * 100000  # Scale price to dollars
    return df, housing

df, housing = load_data()

# Train the Linear Regression Model
X = df.drop(columns=["PRICE"])
y = df["PRICE"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("🏡 California House Price Prediction")
st.write("Enter the house details below to predict the price.")

# User Inputs
col1, col2 = st.columns(2)

with col1:
    med_inc = st.number_input("Median Income ($10,000s)", min_value=0.0, max_value=15.0, value=5.0)
    house_age = st.number_input("House Age", min_value=1.0, max_value=50.0, value=10.0)
    avg_rooms = st.number_input("Average Rooms per House", min_value=1.0, max_value=10.0, value=6.0)
    avg_bedrooms = st.number_input("Average Bedrooms per House", min_value=1.0, max_value=5.0, value=2.0)

with col2:
    population = st.number_input("Population in the Area", min_value=100.0, max_value=50000.0, value=3000.0)
    avg_occupancy = st.number_input("Average Occupancy per Household", min_value=0.5, max_value=10.0, value=3.0) 
    latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=37.0)
    longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-119.0)

# Predict House Price
if st.button("Predict House Price"):
    user_input = np.array([[med_inc, house_age, avg_rooms, avg_bedrooms, population, avg_occupancy, latitude, longitude]])  

    predicted_price = model.predict(user_input)[0]
    
    st.success(f"🏠 Estimated House Price: **${predicted_price:,.2f}**")

    # Model Performance Metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("📊 Model Performance Metrics")
    st.write(f"📌 **Mean Absolute Error (MAE):** {mae:,.2f}")
    st.write(f"📌 **Mean Squared Error (MSE):** {mse:,.2f}")
    st.write(f"📌 **R² Score:** {r2:.2f}")

    # Feature Importance
    st.subheader("📌 Feature Importance")
    feature_importance = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
    feature_importance = feature_importance.sort_values(by="Coefficient", ascending=False)
    st.dataframe(feature_importance)

    # Visualization
    st.subheader("📈 House Price Trend")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices")
    st.pyplot(fig)

st.markdown("""
    <style>
    /* Overall background */
    body {
        background-color: #f8f9fa;
    }

    /* Title styling */
    h1 {
        color: #004d99;
        text-align: center;
        font-weight: bold;
    }

    /* Custom button styling */
    div.stButton > button {
        background-color: #004d99;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 12px 20px;
        border-radius: 12px;
        border: none;
        transition: 0.3s ease-in-out;
    }

    /* Button hover effect */
    div.stButton > button:hover {
        background-color: #002b66;
        transform: scale(1.05);
    }

    /* Input fields */
    div[data-baseweb="input"] > div {
        border-radius: 10px;
        border: 2px solid #004d99 !important;
        padding: 8px;
    }

    /* Style for columns */
    .st-emotion-cache-ocqkz7 {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }

    /* Feature Importance table styling */
    .dataframe {
        border-radius: 10px;
        background-color: white;
        padding: 10px;
    }

    /* Scatter plot style */
    .stPlotlyChart {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }

    </style>
""", unsafe_allow_html=True)
