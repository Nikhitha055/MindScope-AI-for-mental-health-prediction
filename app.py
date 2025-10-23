from flask import Flask, render_template, request
import pickle, json
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load saved artifacts
with open('mental_health_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('symptoms.json', 'r') as f:
    symptom_columns = json.load(f)

@app.route('/')
def home():
    return render_template('index.html', symptoms=symptom_columns)

@app.route('/predict', methods=['POST'])
def predict():
    # Get symptoms from form
    user_symptoms = request.form.getlist('symptoms')

    # Check if no symptoms are selected
    if not user_symptoms:
        error_message = "⚠ Please select at least one symptom before predicting."
        return render_template('index.html', symptoms=symptom_columns, error=error_message)

    # Create input vector
    symptom_input = [1 if symptom in user_symptoms else 0 for symptom in symptom_columns]

    # Scale input
    symptom_input_scaled = scaler.transform([symptom_input])

    # Predict
    prediction_encoded = model.predict(symptom_input_scaled)[0]
    predicted_disorder = label_encoder.inverse_transform([prediction_encoded])[0]

    # Probability scores
    prediction_proba = model.predict_proba(symptom_input_scaled)[0]
    top_3_idx = np.argsort(prediction_proba)[-3:][::-1]
    top_3 = [
        {
            'disorder': label_encoder.classes_[i],
            'probability': round(prediction_proba[i] * 100, 2)
        }
        for i in top_3_idx
    ]

    return render_template(
        'result.html',
        primary=predicted_disorder,
        top_3=top_3,
        symptoms=user_symptoms
    )

if __name__ == '__main__':
    app.run(debug=True)
