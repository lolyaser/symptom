import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("model 2 ai.csv")
    data.columns = data.columns.str.strip()
    data = data.loc[:, ~data.columns.str.contains('^unnamed', case=False)]
    return data

# Train model
@st.cache_resource
def train_model(data):
    features = [col for col in data.columns if col != "Medicines"]
    X = data[features]
    y = data["Medicines"]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)
    return model, le, features

# Predict top medication for each selected symptom
def predict_individual_symptoms(model, le, features, selected_symptoms):
    results = {}
    for sym in selected_symptoms:
        if sym not in features:
            continue
        vec = np.zeros((1, len(features)))
        vec[0, features.index(sym)] = 1
        proba = model.predict_proba(vec)[0]
        top_idx = np.argmax(proba)
        med = le.inverse_transform([top_idx])[0]
        prob = proba[top_idx]
        results[sym] = (med, prob)
    return results

# Streamlit App
def main():
    st.set_page_config(page_title="AI Symptom Checker", layout="centered")
    st.title("🤖 AI Symptom Checker")
    st.write("Select your symptoms, and the AI will suggest the most likely medication for each one.")

    data = load_data()
    model, le, features = train_model(data)

    # Map display names to original feature names
    symptom_display_map = {sym.replace("_", " ").title(): sym for sym in features}
    display_names = list(symptom_display_map.keys())

    selected_display = st.multiselect("📋 Select your symptoms:", display_names)

    if st.button("🔍 Show Medications"):
        if not selected_display:
            st.warning("Please select at least one symptom.")
        else:
            selected_original = [symptom_display_map[sym] for sym in selected_display]
            results = predict_individual_symptoms(model, le, features, selected_original)

            st.subheader("💊 Recommended Medications:")
            for sym, (med, prob) in results.items():
                st.markdown(f"- **{med}** ({prob*100:.1f}%) for symptom: **{sym.replace('_', ' ').title()}**")

if __name__ == "__main__":
    main()
