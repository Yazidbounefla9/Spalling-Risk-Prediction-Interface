import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Charger le modèle
model = joblib.load("C:/Users/Yazid/PycharmProjects/pythonProject1/Spalling/mlp_model.pkl")

# Noms complets et plages des variables
variable_info = {
    "W/B": {"min": 0.186, "max": 0.61, "label": "Water/Binder Ratio"},
    "FA (kg)": {"min": 0, "max": 216, "label": "Fly Ashes (kg)"},
    "L (Kg)": {"min": 0, "max": 126, "label": "Limestone (kg)"},
    "S (Kg)": {"min": 0, "max": 216, "label": "Slag (kg)"},
    "SF (Kg)": {"min": 0, "max": 208, "label": "Silica Fume (kg)"},
    "SF %volume": {"min": 0, "max": 3.026, "label": "Steel Fibers %"},
    "PPF %volume": {"min": 0, "max": 0.645, "label": "Polypropylene Fibers %"},
    "At": {"min": 1, "max": 5, "label": "Aggregate Type 1= Granit 2=Silicious 3= Basalt 4= Non identified 5= Limestone"},
    "CC % Humidité": {"min": 0, "max": 100, "label": "Curing Conditions (% Humidity)"},
    "CD (Days)": {"min": 28, "max": 656, "label": "Curing Duration (Days)"},
    "SV m3": {"min": 0.000125, "max": 4.4814, "label": "Specimen Volume (m3)"},
    "ET (min)": {"min": 2, "max": 600, "label": "Exposure Time (min)"},
    "HR °C/min": {"min": 1, "max": 300, "label": "Heating Rate (°C/min)"},
}

# Interface utilisateur
st.title("Spalling Risk Prediction Interface")

# Entrée utilisateur pour chaque variable
user_inputs = {}
for var, info in variable_info.items():
    user_inputs[var] = st.number_input(
        info["label"],
        min_value=float(info["min"]),
        max_value=float(info["max"]),
        value=(info["max"] + info["min"]) / 2
    )

# Fonction pour normaliser les données
def normalize(value, var_info):
    """Normalise une valeur en utilisant Min-Max scaling."""
    return (value - var_info["min"]) / (var_info["max"] - var_info["min"])

# Conversion des entrées brutes en données normalisées
normalized_data = np.array([
    normalize(user_inputs[var], variable_info[var]) for var in variable_info.keys()
]).reshape(1, -1)

# Prédiction
if st.button("Predict Spalling Risk"):
    prob = model.predict_proba(normalized_data)[0, 1]
    st.write(f"Predicted Probability of Spalling: {prob:.2f}")
