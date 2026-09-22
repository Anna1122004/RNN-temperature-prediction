import streamlit as st
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model("temperature_rnn.keras")

st.title("Temperature Prediction using RNN")

st.write(
    "Enter temperature and vibration values for the previous 2 timestamps."
)

 
st.subheader("Timestamp 1")

temperature_1 = st.number_input(
    "Temperature 1",
    value=80.0
)

vibration_1 = st.number_input(
    "Vibration 1",
    value=3.5
)


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

  
    input_data = np.array([
        [
            [temperature_1, vibration_1],
            [temperature_2, vibration_2]
        ]
    ])

   
    prediction = model.predict(input_data, verbose=0)

    next_temperature = prediction[0][0]

    st.success(
        f"Predicted Next Temperature: {next_temperature:.2f} °C"
    )
