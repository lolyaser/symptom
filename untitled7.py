import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load and clean dataset
@st.cache_data
def load_data():
    data = pd.read_csv('model 2 ai.csv')
    data.columns = data.columns.str.strip()
    data = data.loc[:, ~data.columns.str.contains('^unnamed', case=False)]
    return data

# Train model
def train_model(data):
    features = [col for col in data.columns if col != 'Medicines']
    X = data[features]
    y = data['Medicines']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)
    return model, le, features

# Prediction logic
def predict_medications(model, label_encoder, features, selected_symptoms):
    input_vector = np.zeros((1, len(features)))
    symptom_indices = []

    for symptom in selected_symptoms:
        if symptom in features:
            index = features.index(symptom)
            input_vector[0, index] = 1
            symptom_indices.append(index)

    probas = model.predict_proba(input_vector)[0]
    sorted_indices = np.argsort(probas)[::-1]
    top_preds = [(label_encoder.inverse_transform([i])[0], probas[i]) for i in sorted_indices if probas[i] > 0]

    symptom_map = {}
    for med, prob in top_preds:
        causes = []
        for idx in symptom_indices:
            symptom_vector = np.zeros((1, len(features)))
            symptom_vector[0, idx] = 1
            symptom_proba = model.predict_proba(symptom_vector)[0]
            med_index = np.where(label_encoder.classes_ == med)[0][0]
            if symptom_proba[med_index] > 0:
                causes.append(features[idx].replace("_", " ").title())
        symptom_map[med] = causes

    return top_preds, symptom_map

# Main Streamlit app
def main():
    st.set_page_config(page_title="AI Symptom Checker", layout="wide")
    st.title("💊 AI Symptom Checker")
    st.markdown("Select your symptoms and get recommended medications based on AI prediction.")

    data = load_data()
    model, le, features = train_model(data)

    # Create display-friendly symptom names
    symptom_display_names = [s.replace("_", " ").title() for s in features]
    display_to_internal = {s.replace("_", " ").title(): s for s in features}

    selected_symptoms = st.multiselect("Select your symptoms:", symptom_display_names)

    if st.button("Show Medications"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
            return

        # Convert back to internal names
        selected_original = [display_to_internal[s] for s in selected_symptoms if s in display_to_internal]

        predictions, symptom_map = predict_medications(model, le, features, selected_original)

        st.subheader("Recommended Medications:")

        # One symptom only
        if len(selected_original) == 1:
            shown = 0
            for med, prob in predictions:
                causes = symptom_map.get(med, [])
                st.markdown(f"**{med}** - {prob*100:.1f}% (for {', '.join(causes)})")
                shown += 1
                if shown == 2:
                    break

        # Multiple symptoms
        else:
            # Get top predicted meds for each symptom individually
            individual_top_meds = {}
            for symptom in selected_original:
                vector = np.zeros((1, len(features)))
                vector[0, features.index(symptom)] = 1
                probs = model.predict_proba(vector)[0]
                top_index = np.argmax(probs)
                med = le.inverse_transform([top_index])[0]
                prob = probs[top_index]
                individual_top_meds[med] = (prob, [symptom.replace("_", " ").title()])

            # Intersect meds for all symptoms
            symptom_sets = []
            for symptom in selected_original:
                vector = np.zeros((1, len(features)))
                vector[0, features.index(symptom)] = 1
                probas = model.predict_proba(vector)[0]
                top_indices = np.where(probas > 0)[0]
                meds = set(le.inverse_transform(top_indices))
                symptom_sets.append(meds)

            common_meds = set.intersection(*symptom_sets)

            shown = 0
            if common_meds:
                for med, prob in predictions:
                    if med in common_meds:
                        causes = symptom_map.get(med, [])
                        st.markdown(f"**{med}** - {prob*100:.1f}% (for {', '.join(causes)})")
                        shown += 1
                        if shown == 2:
                            break
            else:
                for med, (prob, causes) in individual_top_meds.items():
                    st.markdown(f"**{med}** - {prob*100:.1f}% (for {', '.join(causes)})")
                    shown += 1
                    if shown == 2:
                        break

if __name__ == "__main__":
    main()
