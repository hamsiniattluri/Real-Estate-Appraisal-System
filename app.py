from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and encoder
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    location = request.form['location']
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])

    # Convert location into number
    location_encoded = encoder.transform([location])[0]

    prediction = model.predict([[location_encoded, area, bedrooms, bathrooms]])

    output = round(prediction[0], 2)

    return render_template(
        'index.html',
        prediction_text=f'Estimated Price: ₹ {output} Lakhs'
    )

if __name__ == "__main__":
    app.run(debug=True)