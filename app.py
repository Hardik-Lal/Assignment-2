from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the saved scaler and model
with open('model.pkl', 'rb') as file:
    scaler, model = pickle.load(file)

# Define the home route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        try:
            inputs = [
                float(request.form['Relative_Compactness']),
                float(request.form['Surface_Area']),
                float(request.form['Wall_Area']),
                float(request.form['Roof_Area']),
                float(request.form['Overall_Height']),
                float(request.form['Orientation']),
                float(request.form['Glazing_Area']),
                float(request.form['Glazing_Area_Distribution'])
            ]
            
            # Preprocess and predict
            inputs_scaled = scaler.transform([inputs])
            prediction = model.predict(inputs_scaled)[0]
            
            return render_template('index.html', prediction=round(prediction, 2))
        except ValueError:
            return render_template('index.html', error="Please enter valid numbers.")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
