from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        Gender = int(request.form['Gender'])
        SeniorCitizen = int(request.form['SeniorCitizen'])
        Tenure = int(request.form['Tenure'])
        InternetService = int(request.form['InternetService'])
        Contract = int(request.form['Contract'])
        PaperlessBilling = int(request.form['PaperlessBilling'])
        MonthlyCharges = float(request.form['MonthlyCharges'])
        TotalCharges = int(request.form['TotalCharges'])

        new_arr = np.array([[Gender, SeniorCitizen, Tenure, InternetService, Contract, PaperlessBilling, MonthlyCharges, TotalCharges]])

        new_output = model.predict(new_arr)
        probabilities = model.predict_proba(new_arr)

        if new_output[0] == 1:
            predicted_result = "Customer is likely to subscribe!"
        else:
            predicted_result = "Customer is unlikely to subscribe"

        confidence = 100*probabilities[0][1]

        return render_template('home.html', predicted_result=predicted_result, confidence=confidence)
    return render_template('home.html', output1="", output2="", query1="", query2="", query3="", query4="", query5="", query6="", query7="", query8="")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
