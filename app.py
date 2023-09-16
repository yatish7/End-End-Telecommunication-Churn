<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Churn Prediction</title>
    <style>
        body {
            background-color: #f0f0f0;
            font-family: Arial, sans-serif;
        }

        .container {
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
            padding: 20px;
            margin-top: 20px;
        }

        .form-group {
            margin-bottom: 15px;
        }

        label {
            font-weight: bold;
        }

        /* Text box style */
        input.small-text-box {
            width: 100%; /* Make text boxes the same width as labels */
            padding: 5px; /* Adjust padding as needed */
            border: 1px solid #ddd;
            border-radius: 5px;
        }

        select.small-select-box {
            width: 100%;
            padding: 5px;
            border-radius: 5px;
        }

        .btn-primary {
            background-color: #007bff;
            color: #fff;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            display: block; /* Make it a block-level element */
            margin: 0 auto; /* Center horizontally */
        }

        .btn-primary:hover {
            background-color: #0056b3;
        }

        .output-group {
            margin-top: 20px;
            padding-top: 20px; /* Added gap between form and prediction box */
            border-top: 1px solid #ddd; /* Added a border above the prediction box */
        }

        .output-group label {
            font-weight: bold;
        }

        /* Text box style for prediction and confidence */
        input.small-text-box {
            width: 100%;
            padding: 5px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }

        h1 {
            text-align: center;
            color: #333;
        }

        /* Style for arranging form groups in rows of 2 */
        .form-row {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
        }

        .form-group {
            flex-basis: calc(50% - 10px);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="row">
            <form action="/" method="POST" class="col-sm-9">
                <h1 align='center'>Telecommunication Churn Model</h1>
                <!-- Row 1 -->
                <div class="form-row">
                    <div class="form-group">
                        <label for="query4">Gender:</label>
                        <select class="form-control small-select-box" id="query4" name="query4" required>
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="query1">SeniorCitizen:</label>
                        <select class="form-control small-select-box" id="query1" name="query1" required>
                            <option value="1">Yes</option>
                            <option value="0">No</option>
                        </select>
                    </div>
                </div>
                <!-- Row 2 -->
                <div class="form-row">
                    <div class="form-group">
                        <label for="query2">Monthly Charges:</label>
                        <input type="text" class="form-control small-text-box" id="query2" name="query2" value="{{ output2 }}">
                    </div>
                    <div class="form-group">
                        <label for="query3">Total Charges:</label>
                        <input type="text" class="form-control small-text-box" id="query3" name="query3" value="{{ output2 }}">
                    </div>
                </div>
                <!-- Row 3 -->
                <div class="form-row">
                    <div class="form-group">
                        <label for="query5">Partner:</label>
                        <select class="form-control small-select-box" id="query5" name="query5" required>
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="query6">Dependents:</label>
                        <select class="form-control small-select-box" id="query6" name="query6" required>
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>
                </div>
                <!-- Row 4 -->
                <div class="form-row">
                    <div class="form-group">
                        <label for="query7">Phone Service:</label>
                        <select class="form-control small-select-box" id="query7" name="query7" required>
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="query8">Multiple Lines:</label>
                        <select class="form-control small-select-box" id="query8" name="query8" required>
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                            <option value="No Phone Service">No Phone Service</option>
                        </select>
                    </div>
                </div>
                <!-- Row 5 -->
                <div class="form-row">
                    <div class="form-group">
                        <label for="query9">Internet Service:</label>
                        <select class="form-control small-select-box" id="query9" name="query9" required>
                            <option value="DSL">DSL</option>
                            <option value="Fiber Optic">Fiber Optic</option>
                            <option value="No">No</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="query10">Online Security:</label>
                        <select class="form-control small-select-box" id="query10" name="query10" required>
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>
                </div>
                <!-- Row 6 -->
                <div class="form-row">
                    <div class="form-group">
                        <label for="query11">Online Backup:</label>
                        <select class="form-control small-select-box" id="query11" name="query11" required>
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="query13">Tech Support:</label>
                        <select class="form-control small-select-box" id="query13" name="query13" required>
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>
                </div>
                <!-- Row 7 -->
                <div class="form-row">
                    <div class="form-group">
                        <label for="query16">Contract:</label>
                        <select class="form-control small-select-box" id="query16" name="query16" required>
                            <option value="Month-to-month">Month-to-month</option>
                            <option value="One year">One year</option>
                            <option value="Two year">Two year</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="query17">Paperless Billing:</label>
                        <select class="form-control small-select-box" id="query17" name="query17" required>
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>
                </div>
                <!-- Row 8 -->
                <div class="form-row">
                    <div class="form-group">
                        <label for="query18">Payment Method:</label>
                        <select class="form-control small-select-box" id="query18" name="query18" required>
                            <option value="Electronic check">Electronic check</option>
                            <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
                            <option value="Credit card (automatic)">Credit card (automatic)</option>
                            <option value="Mailed check">Mailed check</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="query19">Tenure:</label>
                        <input type="text" class="form-control small-text-box" id="query19" name="query19" value="{{ output2 }}">
                    </div>
                </div>
                
                <!-- Centered Submit Button -->
                <div class="form-group text-center">
                    <button type="submit" class="btn btn-primary" name="submit">SUBMIT</button>
                </div>
            </form>
        </div>
    </div>

    <!-- Prediction Box -->
    <div class="container">
            <u><h1 align="center">OUTPUT</h1></u>
            <br>
            <div class="form-row">
                <div class="form-group">
                    <label for="prediction"><u>Prediction:</u></label>
                    <br><br>
                    <textarea class="form-control small-text-box" id="prediction" name="prediction" rows="3" cols="75" autofocus>{{ output1 }}</textarea>
                </div>
                <div class="form-group">
                    <label for="confidence"><u>Confidence:</u></label>
                    <br><br>
                    <textarea class="form-control small-text-box" id="confidence" name="confidence" rows="3" cols="75" autofocus>{{ output2 }}</textarea>
                </div>
            </div>
    </div>
</body>
</html>
