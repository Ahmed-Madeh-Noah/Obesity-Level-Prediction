"""
Model Interface Script
This script defines the function needed to send user input to the classifier
and return the corresponding obesity level.
If the script is run directly, it tests the first column of the cleaned data.

Example Usage:
from Model_Interface import predict_from_input

prediction = predict_from_input(user_input)
"""

import pandas as pd
from Notebooks.Feature_Engineering import add_bmi, add_bmr
import joblib

preprocessor = joblib.load('utils/X_preprocessor.pkl')
model = joblib.load('models/SVC.pkl')
y_encoder = joblib.load('utils/y_encoder.pkl')


def predict_from_input(user_input: pd.Series) -> str:
    user_input["BMI"] = add_bmi(user_input)
    user_input["BMR"] = add_bmr(user_input)
    user_input_df = user_input.to_frame().T
    user_input = preprocessor.transform(user_input_df)
    feature_names = preprocessor.get_feature_names_out()
    clean_feature_names = [name.split("__")[-1] for name in feature_names]
    user_input_df = pd.DataFrame(user_input, columns=clean_feature_names)
    prediction = model.predict(user_input_df)
    return y_encoder.categories_[0][int(prediction[0])]


if __name__ == "__main__":
    first_row = pd.read_csv("data/Obesity_Dataset_Cleaned.csv").iloc[0]
    obesity_level = first_row["Obesity_Level"]
    first_row = first_row.drop("Obesity_Level")
    predict = predict_from_input(first_row)
    print("Row:")
    print(first_row)
    print(f"was predicted to be: {predict}")
    print(f"It is in fact {obesity_level}.")
