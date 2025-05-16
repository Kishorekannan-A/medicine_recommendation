import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset with error handling
try:
    data = pd.read_csv('data/medicine_data.csv')
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit(1)

# Verify column names
expected_columns = ['Age_Category', 'Disease', 'Symptom_1', 'Symptom_2', 'Symptom_3', 'Medicine_Recommendation']
if list(data.columns) != expected_columns:
    print(f"Unexpected columns: {data.columns}")
    exit(1)

# Encode categorical variables
le_age = LabelEncoder()
le_disease = LabelEncoder()
le_symptom1 = LabelEncoder()
le_symptom2 = LabelEncoder()
le_symptom3 = LabelEncoder()
le_medicine = LabelEncoder()

data['Age_Category'] = le_age.fit_transform(data['Age_Category'])
data['Disease'] = le_disease.fit_transform(data['Disease'])
data['Symptom_1'] = le_symptom1.fit_transform(data['Symptom_1'])
data['Symptom_2'] = le_symptom2.fit_transform(data['Symptom_2'])
data['Symptom_3'] = le_symptom3.fit_transform(data['Symptom_3'])
data['Medicine_Recommendation'] = le_medicine.fit_transform(data['Medicine_Recommendation'])

# Features and target
X = data[['Age_Category', 'Disease', 'Symptom_1', 'Symptom_2', 'Symptom_3']]
y = data['Medicine_Recommendation']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('le_age.pkl', 'wb') as f:
    pickle.dump(le_age, f)
with open('le_disease.pkl', 'wb') as f:
    pickle.dump(le_disease, f)
with open('le_symptom1.pkl', 'wb') as f:
    pickle.dump(le_symptom1, f)
with open('le_symptom2.pkl', 'wb') as f:
    pickle.dump(le_symptom2, f)
with open('le_symptom3.pkl', 'wb') as f:
    pickle.dump(le_symptom3, f)
with open('le_medicine.pkl', 'wb') as f:
    pickle.dump(le_medicine, f)

print("Model and encoders saved successfully.")