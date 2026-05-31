from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = "loan_approval_model_new_dataset.pkl"

# Load the trained ML model pipeline
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    model = None
    print(f"ERROR: {MODEL_PATH} not found. Place the .pkl file in the same folder as app.py.")


# These are the exact input columns expected by the trained model.
# Do not include LoanApproved because it is the output/target column.
EXPECTED_COLUMNS = [
    "Age",
    "AnnualIncome",
    "CreditScore",
    "EmploymentStatus",
    "EducationLevel",
    "Experience",
    "LoanAmount",
    "LoanDuration",
    "MaritalStatus",
    "NumberOfDependents",
    "HomeOwnershipStatus",
    "MonthlyDebtPayments",
    "CreditCardUtilizationRate",
    "NumberOfOpenCreditLines",
    "NumberOfCreditInquiries",
    "DebtToIncomeRatio",
    "BankruptcyHistory",
    "LoanPurpose",
    "PreviousLoanDefaults",
    "PaymentHistory",
    "LengthOfCreditHistory",
    "SavingsAccountBalance",
    "CheckingAccountBalance",
    "TotalAssets",
    "TotalLiabilities",
    "MonthlyIncome",
    "UtilityBillsPaymentHistory",
    "JobTenure",
    "NetWorth",
    "TotalDebtToIncomeRatio"
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Loan Approval Prediction API is running.",
        "usage": "Send a POST request to /predict with applicant details as JSON."
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": "Model file not found. Please place loan_approval_model_new_dataset.pkl in the same folder as app.py."
        }), 500

    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "No JSON data received."}), 400

        missing_fields = [col for col in EXPECTED_COLUMNS if col not in data]
        if missing_fields:
            return jsonify({
                "error": "Missing required fields.",
                "missing_fields": missing_fields
            }), 400

        input_data = {col: data[col] for col in EXPECTED_COLUMNS}
        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)[0]

        approval_probability = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0]
            approval_probability = float(probability[1])

        result = "Loan Approved" if int(prediction) == 1 else "Loan Rejected"

        return jsonify({
            "prediction": int(prediction),
            "result": result,
            "approval_probability": approval_probability
        })

    except Exception as e:
        return jsonify({
            "error": "Prediction failed.",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)