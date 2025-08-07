import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__, static_url_path='/static')

# Load your trained ML model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict_form():
    return render_template('predict.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/contact')
def contact():
    return render_template('contacts.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Use lowercase keys to match HTML form names
        gender = request.form['gender'].strip().lower()
        hemoglobin = float(request.form['hemoglobin'])
        mch = float(request.form['mch'])
        mchc = float(request.form['mchc'])
        mcv = float(request.form['mcv'])

        # Convert gender to numeric if your model expects it
        gender_numeric = 1 if gender == 'female' else 0

        # Prepare input
        input_features = np.array([[gender_numeric, hemoglobin, mch, mchc, mcv]])
        columns = ['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV']
        df = pd.DataFrame(input_features, columns=columns)

        # Make prediction
        prediction = model.predict(df)[0]

        # Interpret result
        if prediction == 0:
            result = "You don't have any Anemic Disease"
        elif prediction == 1:
            result = "You have anemic disease"
        else:
            result = "Prediction result unclear"

        return render_template('predict.html', prediction=result)

    except Exception as e:
        return render_template('predict.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
