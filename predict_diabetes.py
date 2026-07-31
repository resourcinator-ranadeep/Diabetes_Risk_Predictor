import pandas as pd
import sklearn as sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
# If Using Logistic Regression approach
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
'''
# 1. Load data efficiently by defining optimal dtypes for 4GB RAM
# (Assuming columns: 'gender', 'age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes')
dtype_dict = {
    'gender': 'category', 
    'age': 'float32', 
    'bmi': 'float32', 
    'HbA1c_level': 'float32', 
    'blood_glucose_level': 'float32', 
    'diabetes': 'int8'
}
df = pd.read_csv('diabetes_prediction_dataset.csv', dtype=dtype_dict)

# 2. Separate Features and Target
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# 3. Handle categorical 'gender' and numeric columns efficiently
numeric_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_features = ['gender']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features) # Drops one column to save memory
    ])

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Process Data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 6. Train a highly efficient model

# If Using Logistic Regression approach
# model = LogisticRegression(solver='saga', max_iter=250, class_weight='balanced') # 'saga' handles large/mid datasets fast


# Using Random Forest Classifier
# We set n_estimators=50 and max_depth=10 to keep your 4GB RAM perfectly safe
# We manually tell the model that a diabetic case is worth roughly 5 times a normal case, 
# instead of the aggressive 11x auto-balance.
custom_weights = {0: 1, 1: 5}
model = RandomForestClassifier(
    n_estimators=50, 
    max_depth=10, 
    class_weight=custom_weights,
    #class_weight='balanced', 
    random_state=42, 
    n_jobs=-1 # Uses all available cores of your CPU to run faster
)
model.fit(X_train_processed, y_train)

# 7. Evaluate
# Using Training Data
predictions = model.predict(X_train_processed)
acc = accuracy_score(y_train, predictions)
print(f"Overall Model Accuracy on training data: {acc:.4f}")

# Using Test Data
predictions = model.predict(X_test_processed)
acc = accuracy_score(y_test, predictions)
print(f"Overall Model Accuracy on test data: {acc:.4f}")

print("-" * 55)
print(classification_report(y_test, predictions))



# Quick Definitions:

# Precision: Out of all the patients the model predicted as diabetic,
# it's the percentage who actually had diabetes (measures quality of positive predictions).

# Recall: Out of all the patients who actually had diabetes,
# it's the percentage the model successfully found (measures quantity of positive cases caught).

# F1-Score: The harmonic balance between Precision and Recall,
# giving you a single metric to judge how well the model handles the diabetic class overall.

# Support: The actual number of true patient records belonging to that specific class in your test dataset (e.g., 1,708 diabetic patients).















import numpy as np

def predict_individual_risk_score(single_patient_data):
    """
    Predicts the diabetes risk score out of 100 for a single patient.
    
    Parameters:
    single_patient_data (dict): Dictionary containing the patient's features.
    
    Example:
    patient = {
        'gender': 'Female',
        'age': 45.0,
        'bmi': 28.4,
        'HbA1c_level': 6.5,
        'blood_glucose_level': 150.0
    }
    """
    # 1. Convert the single dictionary into a 1-row Pandas DataFrame
    # (Matches the original feature names expected by the preprocessor)
    single_df = pd.DataFrame([single_patient_data])
    
    # 2. Match the exact data types to remain memory-efficient and prevent warnings
    single_df = single_df.astype({
        'gender': 'category',
        'age': 'float32',
        'bmi': 'float32',
        'HbA1c_level': 'float32',
        'blood_glucose_level': 'float32'
    })
    
    # 3. Transform the single row using the already-fitted preprocessor
    processed_input = preprocessor.transform(single_df)
    
    # 4. Get the raw prediction probabilities
    # predict_proba returns an array like [[prob_non_diabetic, prob_diabetic]]
    probabilities = model.predict_proba(processed_input)
    
    # Extract the probability for class 1 (Diabetic)
    diabetic_probability = probabilities[0][1]
    
    # 5. Convert to a clean, rounded score out of 100
    risk_score = round(diabetic_probability * 100, 2)
    
    return risk_score

# --- Example Usage ---

# High-risk profile example
high_risk_patient = {
    'gender': 'Male',
    'age': 62.0,
    'bmi': 33.5,
    'HbA1c_level': 7.2,
    'blood_glucose_level': 210.0
}

# Low-risk profile example
low_risk_patient = {
    'gender': 'Female',
    'age': 24.0,
    'bmi': 21.2,
    'HbA1c_level': 4.8,
    'blood_glucose_level': 185.0
}

print(f"High-Risk Patient Score: {predict_individual_risk_score(high_risk_patient)} / 100")
print(f"Low-Risk Patient Score: {predict_individual_risk_score(low_risk_patient)} / 100")



'''