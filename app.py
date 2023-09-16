from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def churn_prediction():
    output1 = ""  # Replace with your actual prediction value
    output2 = ""  # Replace with your actual confidence value

    if request.method == 'POST':
        # Get the form data
        query1 = int(request.form.get('query1'))
        query2 = float(request.form.get('query2'))
        query3 = float(request.form.get('query3'))
        query4 = request.form.get('query4')
        query5 = request.form.get('query5')
        query6 = request.form.get('query6')
        query7 = request.form.get('query7')
        query8 = request.form.get('query8')
        query9 = request.form.get('query9')
        query10 = request.form.get('query10')
        query11 = request.form.get('query11')
        query13 = request.form.get('query13')
        query16 = request.form.get('query16')
        query17 = request.form.get('query17')
        query18 = request.form.get('query18')
        query19 = int(request.form.get('query19'))

        # Perform your churn prediction here based on the form data
        # Replace the following lines with your actual prediction code
        # Example:
        if query1 == 1 and query2 > 50:
            output1 = "Churn: Yes"
            output2 = "Confidence: High"
        else:
            output1 = "Churn: No"
            output2 = "Confidence: Low"

    return render_template('index.html', output1=output1, output2=output2)

if __name__ == '__main__':
    app.run(debug=True)
