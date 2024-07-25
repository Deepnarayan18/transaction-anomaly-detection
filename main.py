from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
with open('anomaly_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the relevant features used during training
relevant_features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs for features
    user_inputs = []
    for feature in relevant_features:
        user_input = float(request.form[feature])
        user_inputs.append(user_input)

    # Create a DataFrame from user inputs
    user_df = pd.DataFrame([user_inputs], columns=relevant_features)

    # Predict anomalies using the model
    user_anomaly_pred = model.predict(user_df)

    # Convert the prediction to binary value (0: normal, 1: anomaly)
    user_anomaly_pred_binary = 1 if user_anomaly_pred == -1 else 0

    if user_anomaly_pred_binary == 1:
        result = "Anomaly detected: This transaction is flagged as an anomaly."
    else:
        result = "No anomaly detected: This transaction is normal."

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
