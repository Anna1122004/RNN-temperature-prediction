import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load model and scalers
model = tf.keras.models.load_model("temperature_rnn.keras")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

st.title("Temperature Prediction using RNN")

st.write(
    "Enter temperature and vibration values for the previous 2 timestamps."
)

# Timestamp 1
st.subheader("Timestamp 1")

temperature_1 = st.number_input(
    "Temperature 1",
    value=80.0
)

vibration_1 = st.number_input(
    "Vibration 1",
    value=3.5
)

# Timestamp 2
st.subheader("Timestamp 2")

temperature_2 = st.number_input(
    "Temperature 2",
    value=83.0
)

vibration_2 = st.number_input(
    "Vibration 2",
    value=3.6
)

if st.button("Predict Next Temperature"):

    # Create input
    new_data = np.array([
        [temperature_1, vibration_1],
        [temperature_2, vibration_2]
    ])

    # Scale input
    new_data_scaled = scaler_X.transform(new_data)

    # Reshape for RNN
    new_data_scaled = new_data_scaled.reshape(1, 2, 2)

    # Predict
    prediction_scaled = model.predict(
        new_data_scaled,
        verbose=0
    )

    # Convert prediction back to original temperature
    prediction = scaler_y.inverse_transform(
        prediction_scaled
    )

    next_temperature = prediction[0][0]

    st.success(
        f"Predicted Next Temperature: {next_temperature:.2f} °C"
    )
