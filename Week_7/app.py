import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("iris_model.pkl")

# App title
st.title("🌸 Iris Flower Species Predictor")

# Sidebar for user input
st.sidebar.header("Input Features")
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length (cm)', 4.3, 7.9, 5.4)
    sepal_width  = st.sidebar.slider('Sepal width (cm)', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 6.9, 1.3)
    petal_width  = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 0.2)

    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Display input
st.subheader("User Input Features")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)

# Mapping target names
target_names = ['Setosa', 'Versicolor', 'Virginica']
predicted_species = target_names[prediction]

# Display prediction
st.subheader("Prediction")
st.write(f"Predicted species: **{predicted_species}**")

# Display probabilities
st.subheader("Prediction Probability")
prob_df = pd.DataFrame(prediction_proba, columns=target_names)
st.write(prob_df)

# Plot probability
st.subheader("Probability Distribution")
fig, ax = plt.subplots()
sns.barplot(x=target_names, y=prediction_proba[0], ax=ax)
ax.set_ylabel("Probability")
ax.set_ylim(0, 1)
st.pyplot(fig)
