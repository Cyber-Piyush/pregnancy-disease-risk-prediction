import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

## 1. Create Synthetic Patient Data
# We will generate 1000 fake patient samples.
np.random.seed(42)
num_samples = 1000

data = {
    'age': np.random.randint(20, 45, size=num_samples),
    'bmi': np.random.uniform(18.0, 45.0, size=num_samples),
    'glucose': np.random.randint(70, 200, size=num_samples),
    'systolic_bp': np.random.randint(90, 180, size=num_samples),
    'diastolic_bp': np.random.randint(60, 110, size=num_samples),
    'hemoglobin': np.random.uniform(9.0, 15.0, size=num_samples) # Key for anaemia
}
df = pd.DataFrame(data)

# Define rules to create the target variable (Disease + Risk Level)
def assign_risk(row):
    # Preeclampsia (High Risk)
    if (row['systolic_bp'] > 160 or row['diastolic_bp'] > 100) and row['bmi'] > 30 and row['age'] > 30:
        return 'preeclampsia_high'
    # Gestational Hypertension (High Risk)
    elif row['systolic_bp'] > 140 or row['diastolic_bp'] > 90:
        return 'hypertension_high'
    # Gestational Diabetes (High Risk)
    elif row['glucose'] > 160 and row['bmi'] > 32:
        return 'diabetes_high'
    # Anaemia (Moderate Risk)
    elif row['hemoglobin'] < 10.5:
        return 'anaemia_moderate'
    # Gestational Diabetes (Moderate Risk)
    elif row['glucose'] > 140:
        return 'diabetes_moderate'
    # Hypertension (Moderate Risk)
    elif row['systolic_bp'] > 130 or row['diastolic_bp'] > 85:
        return 'hypertension_moderate'
    # Anaemia (Low Risk)
    elif row['hemoglobin'] < 11.5:
        return 'anaemia_low'
    # Healthy / Low Risk
    else:
        return 'healthy_low'

# Apply the rules to create our target column
df['target'] = df.apply(assign_risk, axis=1)

print("--- Data Generation Complete ---")
print("Sample of generated data:")
print(df.head())
print("\nRisk category distribution:")
print(df['target'].value_counts())
print("-" * 30)


## 2. Preprocess Data and Train Model

# Define features (X) and target (y)
features = ['age', 'bmi', 'glucose', 'systolic_bp', 'diastolic_bp', 'hemoglobin']
X = df[features]
y = df['target']

# Encode the categorical target variable (e.g., 'anaemia_low') into numbers (e.g., 0)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")
print("Note: High accuracy is expected because the data is fake and rule-based.")
print("-" * 30)


## 3. Create Prediction Function
def predict_patient_risk(patient_data):
    """
    Predicts the disease risk for a single patient.
    
    Args:
    patient_data (dict): A dictionary with the patient's data,
                         including 'name' and all features.
    """
    
    # Extract name for output, remove it from features
    patient_name = patient_data.get('name', 'Unknown')
    age = patient_data.get('age', 0)
    
    # Create a DataFrame from the input data for prediction
    # Ensure the columns are in the same order as the training data
    try:
        input_df = pd.DataFrame([patient_data])
        input_features = input_df[features]
    except KeyError as e:
        print(f"Error: Missing feature in patient data - {e}")
        return

    # Make prediction
    pred_encoded = model.predict(input_features)[0]
    
    # Decode the prediction back to the original label (e.g., "anaemia_moderate")
    pred_label = encoder.inverse_transform([pred_encoded])[0]
    
    # Split the label into disease and risk level
    if '_' in pred_label:
        disease, risk_level = pred_label.split('_')
    else:
        disease = pred_label
        risk_level = "Unknown" # Should not happen

    # Clean up names for final output
    if disease == 'healthy':
        disease = 'No Specific Disease Risk'
    else:
        disease = disease.title() # e.g., "Anaemia"
        
    risk_level = risk_level.title() # e.g., "Moderate"

    # Print the formatted output
    print(f"Patient Name: {patient_name}, Age: {age}")
    print(f"Disease Risk: {disease}")
    print(f"Disease Risk Level: {risk_level}")
    print("-" * 30)


## 4. Run Sample Predictions

print("\n--- Running Sample Predictions ---")

# Sample 1: Alice (Designed to trigger Anaemia)
patient_alice = {
    'name': 'Alice',
    'age': 33,
    'bmi': 24.5,
    'glucose': 95,
    'systolic_bp': 120,
    'diastolic_bp': 75,
    'hemoglobin': 10.2 # Low hemoglobin
}
predict_patient_risk(patient_alice)

# Sample 2: Brenda (Designed to trigger Gestational Diabetes)
patient_brenda = {
    'name': 'Brenda',
    'age': 28,
    'bmi': 33.1,
    'glucose': 170, # High glucose
    'systolic_bp': 125,
    'diastolic_bp': 80,
    'hemoglobin': 12.0
}
predict_patient_risk(patient_brenda)

# Sample 3: Clara (Designed to trigger Preeclampsia/Hypertension)
patient_clara = {
    'name': 'Clara',
    'age': 35,
    'bmi': 31.0,
    'glucose': 110,
    'systolic_bp': 165, # Very high BP
    'diastolic_bp': 102, # Very high BP
    'hemoglobin': 11.8
}
predict_patient_risk(patient_clara)

# Sample 4: Diana (Designed to be Healthy)
patient_diana = {
    'name': 'Diana',
    'age': 26,
    'bmi': 22.0,
    'glucose': 90,
    'systolic_bp': 110,
    'diastolic_bp': 70,
    'hemoglobin': 13.5
}
predict_patient_risk(patient_diana)