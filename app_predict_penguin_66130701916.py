import streamlit as st
import pickle
import pandas as pd

# Load the saved model and encoders
with open('model_penguin_66130701916.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# Create the Streamlit app
st.title('Penguin Species Prediction')

# Input fields for user to enter penguin features
island = st.selectbox('Island', island_encoder.classes_)
culmen_length_mm = st.number_input('Culmen Length (mm)')
culmen_depth_mm = st.number_input('Culmen Depth (mm)')
flipper_length_mm = st.number_input('Flipper Length (mm)')
body_mass_g = st.number_input('Body Mass (g)')
sex = st.selectbox('Sex', sex_encoder.classes_)

# Create a DataFrame from user input
input_data = pd.DataFrame({
    'island': [island],
    'culmen_length_mm': [culmen_length_mm],
    'culmen_depth_mm': [culmen_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g],
    'sex': [sex]
})

# Encode categorical features
input_data['island'] = island_encoder.transform(input_data['island'])
input_data['sex'] = sex_encoder.transform(input_data['sex'])

# Make prediction when user clicks the button
if st.button('Predict'):
    prediction = model.predict(input_data)
    predicted_species = species_encoder.inverse_transform(prediction)
    st.write(f'Predicted Species: **{predicted_species[0]}**')
