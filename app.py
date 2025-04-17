from flask import Flask, request, render_template
import pandas as pd
from Model_Interface import predict_from_input


app = Flask(__name__)


def preprocess_input(form):
    df = pd.DataFrame([{
        "Gender": form["Gender"].lower(),
        "Age": float(form["Age"]),
        "Height": float(form["Height"]),
        "Weight": float(form["Weight"]),
        "Overweight_Family_History": form["family_history_with_overweight"].lower(),
        "High_Calorie_Consumption": form["FAVC"].lower(),
        "Vegetable_Consumption": float(form["FCVC"]),
        "Main_Meals": float(form["NCP"]),
        "Snack_Consumption": form["CAEC"].lower(),
        "Smoker": form["SMOKE"].lower(),
        "Water_Intake": float(form["CH2O"]),
        "Calorie_Monitoring": form["SCC"].lower(),
        "Physical_Activity": float(form["FAF"]),
        "Tech_Time": float(form["TUE"]),
        "Alcohol_Consumption": form["CALC"].lower(),
        "Transportation_Mean": form["MTRANS"].replace(" ", "_")
    }])

    return df


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", prediction_text=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_df = preprocess_input(request.form)
        input_series = input_df.iloc[0]
        prediction = predict_from_input(input_series)

        return render_template("index.html", prediction_text=f"Predicted Obesity Level: {prediction[0].replace('_', ' ')}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True, port=8000)