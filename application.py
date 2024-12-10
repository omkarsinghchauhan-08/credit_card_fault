from flask import Flask, request, render_template
from pymongo import MongoClient
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline
import os

# Initialize Flask application
application = Flask(__name__)
app = application

# MongoDB connection setup using environment variable
mongo_client = MongoClient(os.getenv("mongodb+srv://omkarsinghchauhan2005:<db_password>@cluster0.zi59l.mongodb.net/"))
db = mongo_client['credit_card_faults']  # Database name
collection = db['predictions']  # Collection name

@app.route('/')
def home_page():
    """
    Render the home page with the main interface.
    """
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    """
    Handle form data submission and return predictions.
    """
    if request.method == 'GET':
        # Render the form page for GET requests
        return render_template('form.html')
    else:
        try:
            # Extract and validate form data
            form_data = request.form
            limit_balance = float(form_data.get('LIMIT_BAL'))
            age = int(form_data.get('AGE'))
            gender = int(form_data.get('SEX'))
            education = int(form_data.get('EDUCATION'))
            marital_status = int(form_data.get('MARRIAGE'))
            pay_history = [int(form_data.get(f'PAY_{i}')) for i in range(0, 6)]
            bill_amt = [float(form_data.get(f'BILL_AMT{i}')) for i in range(1, 7)]
            payment_amt = [float(form_data.get(f'PAY_AMT{i}')) for i in range(1, 7)]

            # Map to CustomData and prepare for prediction
            data = CustomData(
                limit_balance=limit_balance,
                age=age,
                gender=gender,
                education=education,
                marital_status=marital_status,
                pay_history=pay_history,
                bill_amt=bill_amt,
                payment_amt=payment_amt,
            )
            final_new_data = data.get_data_as_dataframe()

            # Perform prediction
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(final_new_data)
            result = "Fault" if pred[0] == 1 else "No Fault"

            # Save the input and prediction to MongoDB
            prediction_record = {
                "input_data": final_new_data.to_dict(orient='records')[0],
                "prediction": result
            }
            collection.insert_one(prediction_record)

            # Render form.html with result
            return render_template('form.html', final_result=result)

        except Exception as e:
            # Render form.html with error message
            return render_template('form.html', error_message=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
