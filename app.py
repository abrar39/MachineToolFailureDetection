from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
with open('models/best_lof_model.pkl', 'rb') as f:
    model = pickle.load(f)

#with open('scaler.pkl', 'rb') as f:
#    scaler = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    inputs = None

    if request.method == 'POST':
        if request.form.get('action') == 'predict':
            try:
                # Get user inputs
                process_temp = float(request.form['process_temperature'])
                air_temp = float(request.form['air_temperature'])
                rotational_speed = float(request.form['rotational_speed'])
                torque = float(request.form['torque'])
                tool_wear = float(request.form['tool_wear'])
                type_input = request.form['type'].strip().upper()

                # Validate type input
                if type_input not in ['H', 'M', 'L']:
                    raise ValueError("Invalid type. Enter H, M, or L.")

                # One-hot encoding for 'Type'
                type_h = 1 if type_input == 'H' else 0
                type_m = 1 if type_input == 'M' else 0
                type_l = 1 if type_input == 'L' else 0

                # Construct input in correct order
                columns = [
                    'Air temperature', 
                    'Process temperature', 
                    'Rotational speed',
                    'Torque', 
                    'Tool wear', 
                    'Type_H', 
                    'Type_L', 
                    'Type_M'
                ]

                raw_features = pd.DataFrame([[air_temp, process_temp, rotational_speed, torque, tool_wear, type_h, type_l, type_m]], columns=columns)

                # Apply StandardScaler
                #scaled_features = scaler.transform(raw_features)

                # Predict using the trained model
                pred = model.predict(raw_features)

                prediction = "Failure" if pred[0] == -1 else "Normal"

                inputs = {                    
                    'Air temperature': air_temp,
                    'Process temperature': process_temp,
                    'Rotational speed': rotational_speed,
                    'Torque': torque,
                    'Tool wear': tool_wear,
                    'Type': type_input
                }

            except ValueError as e:
                prediction = f"Invalid input: {str(e)}"

        elif request.form.get('action') == 'reset':
            return render_template('index.html')

    return render_template('index.html', prediction=prediction, inputs=inputs)


if __name__ == '__main__':
    app.run(debug=True)
