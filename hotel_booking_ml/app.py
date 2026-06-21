import streamlit as st
import pandas as pd
import json
from catboost import CatBoostClassifier

@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("catboost_hotel_model.cbm")
    return model

model = load_model()

with open("ui_cat_values.json", "r") as f:
    UI = json.load(f)

FEATURE_ORDER = model.feature_names_
CAT_FEATURES = [
    "type_of_meal_plan",
    "room_type_reserved",
    "market_segment_type"
]


def build_input():
    user = {
        "no_of_adults": st.number_input("Adults", 1, 10, 2),
        "no_of_children": st.number_input("Children", 0, 5, 0),
        "no_of_weekend_nights": st.number_input("Weekend nights", 0, 7, 1),
        "no_of_week_nights": st.number_input("Week nights", 0, 14, 2),

        "type_of_meal_plan": st.selectbox(
            "Meal plan",
            UI["type_of_meal_plan"]
        ),

        "room_type_reserved": st.selectbox(
            "Room type",
            UI["room_type_reserved"]
        ),

        "market_segment_type": st.selectbox(
            "Market segment",
            UI["market_segment_type"]
        ),

        "required_car_parking_space": int(st.checkbox("Parking")),
        "no_of_special_requests": st.slider("Requests", 0, 5, 0)
    }

    date = st.date_input("Arrival date")

    auto = {
        "lead_time": 30.0,
        "arrival_year": date.year,
        "arrival_month": date.month,
        "arrival_date": date.day,
        "avg_price_per_room": 5000.0,
        "repeated_guest": 0,
        "no_of_previous_cancellations": 0,
        "no_of_previous_bookings_not_canceled": 0
    }

    return {**user, **auto}

st.title("Hotel Booking Prediction")

input_data = build_input()
df = pd.DataFrame([input_data])
for col in CAT_FEATURES:
    df[col] = df[col].astype(str)
df = df[FEATURE_ORDER]

st.write("Input data")
st.dataframe(df)

if st.button("Predict"):
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0]

    st.subheader("Result")
    st.write("Prediction:", pred)
    st.write("Class 0:", proba[0])
    st.write("Class 1:", proba[1])

