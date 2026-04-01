from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)


# Home route
@app.route("/")
def home():
    return render_template("frontend.html")  # you can rename to index.html if needed


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Convert numeric columns
        numeric_cols = [
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History"
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])

        # Feature Engineering (MUST match training)
        df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
        df["Income_Loan_Ratio"] = df["TotalIncome"] / (df["LoanAmount"] + 1)
        df["EMI"] = df["LoanAmount"] / (df["Loan_Amount_Term"] + 1)

        # Prediction
        proba = model.predict_proba(df)[0][1]

        # Custom threshold
        prediction = "Loan Approved " if proba > 0.6 else "Loan Not Approved "

        return render_template(
            "result.html",
            prediction=prediction,
            probability=round(float(proba), 3),
            previous_values=data
        )

    except Exception as e:
        return f"Error occurred: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)