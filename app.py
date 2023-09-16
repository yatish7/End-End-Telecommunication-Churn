# coding: utf-8

import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask("__name__")

df_1 = pd.read_csv("tel_churn.csv")

q = ""

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    try:
        # Extract input values from the form
        inputQuery1 = request.form['query1']
        inputQuery2 = request.form['query2']
        inputQuery3 = request.form['query3']
        inputQuery4 = request.form['query4']
        inputQuery5 = request.form['query5']
        inputQuery6 = request.form['query6']
        inputQuery7 = request.form['query7']
        inputQuery8 = request.form['query8']
        inputQuery9 = request.form['query9']
        inputQuery10 = request.form['query10']
        inputQuery11 = request.form['query11']
        inputQuery13 = request.form['query13']
        inputQuery16 = request.form['query16']
        inputQuery17 = request.form['query17']
        inputQuery18 = request.form['query18']
        inputQuery19 = request.form['query19']

        # Check if input values are valid
        def is_valid_float(value):
            try:
                float(value)
                return True
            except ValueError:
                return False

        input_fields = [
            inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5,
            inputQuery6, inputQuery7, inputQuery8, inputQuery9, inputQuery10,
            inputQuery11, inputQuery13, inputQuery16, inputQuery17, inputQuery18, inputQuery19
        ]

        if not all(map(is_valid_float, input_fields)):
            return render_template('home.html', error_message="Invalid input. Please provide valid numeric values.")

        # Load the model
        model = pickle.load(open("model.sav", "rb"))

        # Create a DataFrame from the input data
        input_data = pd.DataFrame([input_fields], columns=[
            'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
            'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'TechSupport',
            'Contract', 'PaperlessBilling',
            'PaymentMethod', 'tenure'
        ])

        # Process the input data and make predictions
        df_2 = pd.concat([df_1, input_data], ignore_index=True)
        # Group the tenure in bins of 12 months
        labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
        df_2['tenure_group'] = pd.cut(df_2.tenure.astype(float), range(1, 80, 12), right=False, labels=labels)
        # Drop column customerID and tenure
        df_2.drop(columns=['tenure'], axis=1, inplace=True)
        new_df__dummies = pd.get_dummies(df_2[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                                'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                                'TechSupport',
                                                'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']])
        single = model.predict(new_df__dummies.tail(1))
        probability = model.predict_proba(new_df__dummies.tail(1))[:, 1]
        if single == 1:
            o1 = "This customer is likely to be churned!!"
            o2 = "Confidence: {}".format(probability * 100)
        else:
            o1 = "This customer is likely to continue!!"
            o2 = "Confidence: {}".format(probability * 100)
        return render_template('home.html', output1=o1, output2=o2, query1=inputQuery1, query2=inputQuery2,
                               query3=inputQuery3, query4=inputQuery4, query5=inputQuery5, query6=inputQuery6,
                               query7=inputQuery7, query8=inputQuery8, query9=inputQuery9, query10=inputQuery10,
                               query11=inputQuery11, query13=inputQuery13, query16=inputQuery16, query17=inputQuery17,
                               query18=inputQuery18, query19=inputQuery19)
    except Exception as e:
        return render_template('home.html', error_message=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
