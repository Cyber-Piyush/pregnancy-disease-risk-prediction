import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
data_path = "synthetic_pregnancy_disease_data_with_normals.csv"
df = pd.read_csv(data_path)

# Define feature prioritization (higher weights for important features)
feature_weights = {
    "Blood_Sugar": 3,
    "Blood_Pressure_Systolic": 2,
    "Blood_Pressure_Diastolic": 2,
    "BMI": 2,
    "Hemoglobin": 2,
    "Age": 1,
    "Hydration_Level": 1,
    "Diet_Quality": 1,
    "History_of_Disease": 2,
    "Iron_Intake": 1,
    "Hormone_Levels": 2,
    "Number_of_Pregnancies": 1,
    "Nausea_Intensity": 1,
    "Presence_of_Pain": 1,
    "Fetal_Weight": 2,
    "Amniotic_Fluid_Index": 2,
    "Smoking_Status": 1
}

# Preprocessing
def preprocess_data(df):
    # Encode categorical features
    categorical_cols = ["Protein_in_Urine", "Hydration_Level", "Diet_Quality", 
                        "History_of_Disease", "Nausea_Intensity", "Presence_of_Pain", 
                        "Smoking_Status", "Disease", "Risk_Level"]

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols.remove("Disease")
    numerical_cols.remove("Risk_Level")
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df, label_encoders, scaler, numerical_cols

df, label_encoders, scaler, numerical_cols = preprocess_data(df)

# Split the dataset into features and labels
X = df.drop(columns=["Disease", "Risk_Level"])
Y_disease = df["Disease"]
Y_risk = df["Risk_Level"]

# Train-test split
X_train, X_test, Y_disease_train, Y_disease_test = train_test_split(X, Y_disease, test_size=0.2, random_state=42)
_, _, Y_risk_train, Y_risk_test = train_test_split(X, Y_risk, test_size=0.2, random_state=42)

# Apply feature prioritization (weights)
for feature, weight in feature_weights.items():
    if feature in X_train.columns:
        X_train[feature] *= weight
        X_test[feature] *= weight

# Train a Random Forest Classifier for disease prediction
rf_disease = RandomForestClassifier(random_state=42)
rf_disease.fit(X_train, Y_disease_train)

# Train a Random Forest Classifier for risk level prediction
rf_risk = RandomForestClassifier(random_state=42)
rf_risk.fit(X_train, Y_risk_train)

# Evaluate the models
Y_disease_pred = rf_disease.predict(X_test)
Y_risk_pred = rf_risk.predict(X_test)

print("Disease Prediction Report:\n", classification_report(Y_disease_test, Y_disease_pred))
print("Risk Level Prediction Report:\n", classification_report(Y_risk_test, Y_risk_pred))

# Define a function to take new user input and make predictions, handling missing values
def predict_disease_and_risk_with_missing(user_input):
    """
    Predict the disease and risk level based on user input, handling missing values.

    Parameters:
        user_input (dict): Dictionary of user inputs with feature values.

    Returns:
        dict: Predicted disease and risk level.
    """
    # Define default values for missing features
    default_values = {
        "Age": 25,
        "BMI": 22,
        "Blood_Pressure_Systolic": 120,
        "Blood_Pressure_Diastolic": 80,
        "Blood_Sugar": 100,
        "Hemoglobin": 12,
        "Protein_in_Urine": "Negative",
        "Hydration_Level": "Normal",
        "Diet_Quality": "Good",
        "History_of_Disease": "No",
        "Iron_Intake": 18,
        "Hormone_Levels": 7,
        "Number_of_Pregnancies": 1,
        "Nausea_Intensity": "Mild",
        "Presence_of_Pain": "No",
        "Fetal_Weight": 2500,
        "Amniotic_Fluid_Index": 10,
        "Smoking_Status": "No"
    }

    # Fill missing values with defaults
    complete_input = {}
    for key, default in default_values.items():
        if key in user_input and user_input[key]:
            try:
                if key in numerical_cols:
                    complete_input[key] = float(user_input[key])
                else:
                    complete_input[key] = user_input[key]
            except ValueError:
                complete_input[key] = default
        else:
            complete_input[key] = default

    # Convert user input into a DataFrame
    input_df = pd.DataFrame([complete_input])

    # Encode categorical variables
    for col, le in label_encoders.items():
        if col in input_df:
            input_df[col] = le.transform(input_df[col])

    # Scale numerical variables
    for col in numerical_cols:
        if col in input_df:
            input_df[col] = (input_df[col] - scaler.mean_[numerical_cols.index(col)]) / scaler.scale_[numerical_cols.index(col)]

    # Apply feature weights
    for feature, weight in feature_weights.items():
        if feature in input_df:
            input_df[feature] *= weight

    # Make predictions
    predicted_disease = rf_disease.predict(input_df)[0]
    predicted_risk = rf_risk.predict(input_df)[0]

    # Decode predictions
    predicted_disease = label_encoders["Disease"].inverse_transform([predicted_disease])[0]
    predicted_risk = label_encoders["Risk_Level"].inverse_transform([predicted_risk])[0]

    return {"Predicted Disease": predicted_disease, "Predicted Risk Level": predicted_risk}

# Interactive input
def get_user_input():
    features = [
        "Age", "BMI", "Blood_Pressure_Systolic", "Blood_Pressure_Diastolic",
        "Blood_Sugar", "Hemoglobin", "Protein_in_Urine", "Hydration_Level",
        "Diet_Quality", "History_of_Disease", "Iron_Intake", "Hormone_Levels",
        "Number_of_Pregnancies", "Nausea_Intensity", "Presence_of_Pain",
        "Fetal_Weight", "Amniotic_Fluid_Index", "Smoking_Status"
    ]
    user_input = {}
    for feature in features:
        value = input(f"Enter {feature} (or press Enter to skip): ")
        user_input[feature] = value if value else None
    return user_input

# Get user input and predict
new_input = get_user_input()
prediction = predict_disease_and_risk_with_missing(new_input)
print("Prediction:", prediction)

