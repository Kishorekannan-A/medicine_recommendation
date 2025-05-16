from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load model and encoders
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('le_age.pkl', 'rb') as f:
        le_age = pickle.load(f)
    with open('le_disease.pkl', 'rb') as f:
        le_disease = pickle.load(f)
    with open('le_symptom1.pkl', 'rb') as f:
        le_symptom1 = pickle.load(f)
    with open('le_symptom2.pkl', 'rb') as f:
        le_symptom2 = pickle.load(f)
    with open('le_symptom3.pkl', 'rb') as f:
        le_symptom3 = pickle.load(f)
    with open('le_medicine.pkl', 'rb') as f:
        le_medicine = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error: Missing model or encoder file - {e}")
    exit(1)

# Load dataset for disease and symptom lists
try:
    data = pd.read_csv('data/medicine_data.csv')
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit(1)

diseases = sorted(data['Disease'].unique())

# Symptom mappings
symptom_map = {
    'Fever': {'Symptom_1': ['High temp (102+)', 'Mild temp (99-100)'], 'Symptom_2': ['3+ days', '1-2 days'], 'Symptom_3': ['Chills', 'No chills']},
    'Vomiting': {'Symptom_1': ['5+ times', '2-3 times'], 'Symptom_2': ['Dehydration', 'No dehydration'], 'Symptom_3': ['Nausea', 'Mild nausea']},
    'Common Cold': {'Symptom_1': ['Runny nose', 'No runny nose'], 'Symptom_2': ['Dry cough', 'Wet cough'], 'Symptom_3': ['Fever', 'No fever']},
    'Skin Infection': {'Symptom_1': ['Red', 'Black spot'], 'Symptom_2': ['Arm', 'Foot'], 'Symptom_3': ['Itching', 'No itching']},
    'Diarrhea': {'Symptom_1': ['5+ times', '2-3 times'], 'Symptom_2': ['Dehydration', 'No dehydration'], 'Symptom_3': ['Pain', 'No pain']},
    'Headache': {'Symptom_1': ['2+ hours', '1 hour'], 'Symptom_2': ['High intensity', 'Low intensity'], 'Symptom_3': ['No blur', 'Vision blur']},
    'Acidity': {'Symptom_1': ['Burning', 'No burning'], 'Symptom_2': ['Bloating', 'No bloating'], 'Symptom_3': ['Nausea', 'No nausea']},
    'Eye Infection': {'Symptom_1': ['Redness', 'No redness'], 'Symptom_2': ['Discharge', 'No discharge'], 'Symptom_3': ['Itching', 'No itching']},
    'Cough': {'Symptom_1': ['Dry', 'Wet'], 'Symptom_2': ['3+ days', '1-2 days'], 'Symptom_3': ['Fever', 'No fever']},
    'Body Pain': {'Symptom_1': ['Back', 'Leg'], 'Symptom_2': ['High intensity', 'Low intensity'], 'Symptom_3': ['Fever', 'No fever']},
    'Allergies': {'Symptom_1': ['Dust', 'Pollen'], 'Symptom_2': ['Rash', 'No rash'], 'Symptom_3': ['Sneezing', 'No sneezing']},
    'Ear Infection': {'Symptom_1': ['Pain', 'No pain'], 'Symptom_2': ['Discharge', 'No discharge'], 'Symptom_3': ['No hearing loss', 'Hearing loss']},
    'Sore Throat': {'Symptom_1': ['Pain', 'No pain'], 'Symptom_2': ['Fever', 'No fever'], 'Symptom_3': ['Swallowing difficulty', 'No difficulty']},
    'Fatigue': {'Symptom_1': ['5+ days', '2 days'], 'Symptom_2': ['Poor sleep', 'Good sleep'], 'Symptom_3': ['Weakness', 'No weakness']},
    'Joint Pain': {'Symptom_1': ['Knee', 'Shoulder'], 'Symptom_2': ['Swelling', 'No swelling'], 'Symptom_3': ['Stiffness', 'No stiffness']},
    'Stomach Ache': {'Symptom_1': ['Upper', 'Lower'], 'Symptom_2': ['Nausea', 'No nausea'], 'Symptom_3': ['Diarrhea', 'No diarrhea']},
    'Toothache': {'Symptom_1': ['High pain', 'Low pain'], 'Symptom_2': ['Swelling', 'No swelling'], 'Symptom_3': ['Sensitivity', 'No sensitivity']},
    'Constipation': {'Symptom_1': ['3+ days', '1-2 days'], 'Symptom_2': ['Pain', 'No pain'], 'Symptom_3': ['Bloating', 'No bloating']},
    'Indigestion': {'Symptom_1': ['Bloating', 'No bloating'], 'Symptom_2': ['Nausea', 'No nausea'], 'Symptom_3': ['Burning', 'No burning']},
    'Malaria': {'Symptom_1': ['Fever', 'No fever'], 'Symptom_2': ['Chills', 'No chills'], 'Symptom_3': ['Sweating', 'No sweating']},
    'Dengue': {'Symptom_1': ['Fever', 'No fever'], 'Symptom_2': ['Rash', 'No rash'], 'Symptom_3': ['Joint pain', 'No joint pain']},
    'Typhoid': {'Symptom_1': ['Fever', 'No fever'], 'Symptom_2': ['Weakness', 'No weakness'], 'Symptom_3': ['Pain', 'No pain']},
    'Jaundice': {'Symptom_1': ['Yellowing', 'No yellowing'], 'Symptom_2': ['Fatigue', 'No fatigue'], 'Symptom_3': ['Nausea', 'No nausea']},
    'Asthma': {'Symptom_1': ['Wheezing', 'No wheezing'], 'Symptom_2': ['Shortness of breath', 'No shortness'], 'Symptom_3': ['Cough', 'No cough']},
    'Sinusitis': {'Symptom_1': ['Congestion', 'No congestion'], 'Symptom_2': ['Pain', 'No pain'], 'Symptom_3': ['Fever', 'No fever']},
    'Urinary Infection': {'Symptom_1': ['Burning', 'No burning'], 'Symptom_2': ['5+ times', '2-3 times'], 'Symptom_3': ['Cloudy urine', 'No cloudy urine']},
    'Back Pain': {'Symptom_1': ['Lower', 'Upper'], 'Symptom_2': ['2+ days', '1 day'], 'Symptom_3': ['Stiffness', 'No stiffness']},
    'Nasal Congestion': {'Symptom_1': ['3+ days', '1-2 days'], 'Symptom_2': ['Fever', 'No fever'], 'Symptom_3': ['Headache', 'No headache']},
    'Throat Infection': {'Symptom_1': ['Pain', 'No pain'], 'Symptom_2': ['Fever', 'No fever'], 'Symptom_3': ['Cough', 'No cough']},
    'Piles': {'Symptom_1': ['Pain', 'No pain'], 'Symptom_2': ['Bleeding', 'No bleeding'], 'Symptom_3': ['Itching', 'No itching']},
    'Migraine': {'Symptom_1': ['High pain', 'Low pain'], 'Symptom_2': ['Nausea', 'No nausea'], 'Symptom_3': ['Light sensitivity', 'No sensitivity']},
    'Gastritis': {'Symptom_1': ['Burning', 'No burning'], 'Symptom_2': ['Nausea', 'No nausea'], 'Symptom_3': ['Bloating', 'No bloating']},
    'Bronchitis': {'Symptom_1': ['Cough', 'No cough'], 'Symptom_2': ['Fever', 'No fever'], 'Symptom_3': ['Chest pain', 'No chest pain']},
    'Heat Rash': {'Symptom_1': ['Itching', 'No itching'], 'Symptom_2': ['Redness', 'No redness'], 'Symptom_3': ['Sweating', 'No sweating']},
    'Conjunctivitis': {'Symptom_1': ['Redness', 'No redness'], 'Symptom_2': ['Itching', 'No itching'], 'Symptom_3': ['Discharge', 'No discharge']},
    'Tonsillitis': {'Symptom_1': ['Pain', 'No pain'], 'Symptom_2': ['Fever', 'No fever'], 'Symptom_3': ['Swallowing difficulty', 'No difficulty']},
    'Muscle Cramps': {'Symptom_1': ['Leg', 'Arm'], 'Symptom_2': ['1+ hour', 'Less than 1 hour'], 'Symptom_3': ['High pain', 'Low pain']},
    'Food Poisoning': {'Symptom_1': ['Nausea', 'No nausea'], 'Symptom_2': ['Vomiting', 'No vomiting'], 'Symptom_3': ['Diarrhea', 'No diarrhea']},
    'Cholera': {'Symptom_1': ['Diarrhea', 'No diarrhea'], 'Symptom_2': ['Dehydration', 'No dehydration'], 'Symptom_3': ['Vomiting', 'No vomiting']},
    'Chickenpox': {'Symptom_1': ['Rash', 'No rash'], 'Symptom_2': ['Fever', 'No fever'], 'Symptom_3': ['Itching', 'No itching']},
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        age = request.form.get('age')
        if age and age.isdigit() and int(age) > 0:
            return redirect(url_for('questions', age=age, step=1))
        return render_template('index.html', error="Please enter a valid age")
    return render_template('index.html')

@app.route('/questions', methods=['GET', 'POST'])
def questions():
    age = request.args.get('age', '')
    step = int(request.args.get('step', 1))
    disease = request.args.get('disease', '')
    symptom1 = request.args.get('symptom1', '')
    symptom2 = request.args.get('symptom2', '')

    if not age:
        return redirect(url_for('index'))

    if request.method == 'POST':
        if step == 1:
            disease = request.form.get('disease')
            if not disease or disease not in diseases:
                return render_template('questions.html', step=1, diseases=diseases, age=age, error="Please select a valid disease")
            return redirect(url_for('questions', age=age, step=2, disease=disease))
        elif step == 2:
            disease = request.form.get('disease')
            symptom1 = request.form.get('symptom1')
            symptoms1 = symptom_map.get(disease, {}).get('Symptom_1', [])
            if not disease or not symptom1 or symptom1 not in symptoms1:
                return render_template('questions.html', step=2, symptoms1=symptoms1, disease=disease or '', age=age, error="Please select a valid symptom")
            return redirect(url_for('questions', age=age, step=3, disease=disease, symptom1=symptom1))
        elif step == 3:
            disease = request.form.get('disease')
            symptom1 = request.form.get('symptom1')
            symptom2 = request.form.get('symptom2')
            symptoms2 = symptom_map.get(disease, {}).get('Symptom_2', [])
            if not disease or not symptom1 or not symptom2 or symptom2 not in symptoms2:
                return render_template('questions.html', step=3, symptoms2=symptoms2, disease=disease or '', symptom1=symptom1 or '', age=age, error="Please select a valid symptom")
            return redirect(url_for('questions', age=age, step=4, disease=disease, symptom1=symptom1, symptom2=symptom2))
        elif step == 4:
            disease = request.form.get('disease')
            symptom1 = request.form.get('symptom1')
            symptom2 = request.form.get('symptom2')
            symptom3 = request.form.get('symptom3')
            symptoms3 = symptom_map.get(disease, {}).get('Symptom_3', [])
            if not disease or not symptom1 or not symptom2 or not symptom3 or symptom3 not in symptoms3:
                return render_template('questions.html', step=4, symptoms3=symptoms3, disease=disease or '', symptom1=symptom1 or '', symptom2=symptom2 or '', age=age, error="Please select a valid symptom")
            try:
                age = int(age)
                age_category = 'Below 16' if age < 16 else '16-60' if age <= 60 else 'Above 60'
                age_encoded = le_age.transform([age_category])[0]
                disease_encoded = le_disease.transform([disease])[0]
                symptom1_encoded = le_symptom1.transform([symptom1])[0]
                symptom2_encoded = le_symptom2.transform([symptom2])[0]
                symptom3_encoded = le_symptom3.transform([symptom3])[0]
                input_data = np.array([[age_encoded, disease_encoded, symptom1_encoded, symptom2_encoded, symptom3_encoded]])
                prediction = model.predict(input_data)
                medicine = le_medicine.inverse_transform(prediction)[0]
                return render_template('result.html', medicine=medicine, age=age, disease=disease, symptom1=symptom1, symptom2=symptom2, symptom3=symptom3)
            except Exception as e:
                return render_template('result.html', error=f"Error: {str(e)}")

    if step == 1:
        return render_template('questions.html', step=step, diseases=diseases, age=age)
    elif step == 2:
        if not disease:
            return redirect(url_for('questions', age=age, step=1))
        symptoms1 = symptom_map.get(disease, {}).get('Symptom_1', [])
        return render_template('questions.html', step=step, symptoms1=symptoms1, disease=disease, age=age)
    elif step == 3:
        if not disease:
            return redirect(url_for('questions', age=age, step=1))
        if not symptom1:
            return redirect(url_for('questions', age=age, step=2, disease=disease))
        symptoms2 = symptom_map.get(disease, {}).get('Symptom_2', [])
        return render_template('questions.html', step=step, symptoms2=symptoms2, disease=disease, symptom1=symptom1, age=age)
    elif step == 4:
        if not disease:
            return redirect(url_for('questions', age=age, step=1))
        if not symptom1:
            return redirect(url_for('questions', age=age, step=2, disease=disease))
        if not symptom2:
            return redirect(url_for('questions', age=age, step=3, disease=disease, symptom1=symptom1))
        symptoms3 = symptom_map.get(disease, {}).get('Symptom_3', [])
        return render_template('questions.html', step=step, symptoms3=symptoms3, disease=disease, symptom1=symptom1, symptom2=symptom2, age=age)

if __name__ == '__main__':
    app.run(debug=True)