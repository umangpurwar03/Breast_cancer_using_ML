import streamlit as st
import joblib
import numpy as np

# Load the trained RandomForest model
rf_model = joblib.load(r'rf1.joblib')

# Define the features
features = ['concavity_mean', 'concavity_worst', 'radius_mean', 
                     'radius_se', 'texture_mean', 'smoothness_worst', 
                     'compactness_se', 'fractal_dimension_worst', 
                     'smoothness_mean', 'concavity_se']

def predict_cancer(features):
    # Prepare the input data as a numpy array
    input_data = np.array(features).reshape(1, -1)
    # Make prediction
    prediction = rf_model.predict(input_data)
    # Convert prediction to string
    cancer_type = 'Malignant' if prediction[0] == 1 else 'Benign'
    return cancer_type

# Streamlit UI
def main():
    st.title("Breast Cancer Prediction")
    
    # Input fields for the features
    st.sidebar.title("Input Features")
    input_features = {}
    for feature in features:
        input_features[feature] = st.sidebar.number_input(f"Enter {feature}", step=0.01)

    # Predict
    if st.button("Predict"):
        feature_values = [input_features[feature] for feature in features]
        result = predict_cancer(feature_values)
        st.write("Prediction:", result)

if __name__ == '__main__':
    main()
