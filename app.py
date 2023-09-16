# Import necessary libraries
import pandas as pd
from flask import Flask, request, render_template
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load your pre-trained model
model = pickle.load(open("model.sav", "rb"))

# Load your dataset
df_1 = pd.read_csv("first_telc.csv")

@app.route("/")
def loadPage():
    return render_template('home.html', output1="", output2="", query1="", query2="", query3="", query4="", query5="", query6="", query7="", query8="", query9="", query10="", query11="", query12="", query13="", query14="", query15="", query16="", query17="", query18="", query19="")

@app.route("/", methods=['POST'])
def predict():
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
    inputQuery12 = request.form['query12']
    inputQuery13 = request.form['query13']
    inputQuery14 = request.form['query14']
    inputQuery15 = request.form['query15']
    inputQuery16 = request.form['query16']
    inputQuery17 = request.form['query17']
    inputQuery18 = request.form['query18']
    inputQuery19 = request.form['query19']

    # Create a new DataFrame with the input data
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7,
             inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,
             inputQuery15, inputQuery16, inputQuery17, inputQuery18, inputQuery19]]

    new_df = pd.DataFrame(data, columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
                                         'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                         'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                         'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                         'PaymentMethod', 'tenure'])

    # Concatenate the new data with the original dataset
    df_2 = pd.concat([df_1, new_df], ignore_index=True)

    # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)

    # Drop unnecessary columns
    df_2.drop(columns=['tenure'], axis=1, inplace=True)

    # Perform one-hot encoding on categorical variables
    new_df_dummies = pd.get_dummies(df_2[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                          'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                          'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                          'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']])

    # Predict using the loaded model
    single = model.predict(new_df_dummies.tail(1))
    probability = model.predict_proba(new_df_dummies.tail(1))[:, 1]

    # Prepare output messages
    if single == 1:
        output1 = "This customer is likely to be churned!!"
    else:
        output1 = "This customer is likely to continue!!"

    output2 = "Confidence: {}%".format(probability * 100)

    # Render the template with the output messages and input values
    return render_template('home.html', output1=output1, output2=output2,
                           query1=inputQuery1, query2=inputQuery2, query3=inputQuery3, query4=inputQuery4, query5=inputQuery5,
                           query6=inputQuery6, query7=inputQuery7, query8=inputQuery8, query9=inputQuery9, query10=inputQuery10,
                           query11=inputQuery11, query12=inputQuery12, query13=inputQuery13, query14=inputQuery14,
                           query15=inputQuery15, query16=inputQuery16, query17=inputQuery17, query18=inputQuery18, query19=inputQuery19)

# Run the app if executed as the main script
if __name__ == "__main__":
    app.run()
