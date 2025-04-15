"""
This script defines the functions that calculates the Body Mass Index and Basal Metabolic Rate
If the script is run directly, it adds those features to the cleaned data and saves the engineered data.
"""

import pandas as pd


def add_bmi(row: pd.Series) -> float:
    bmi = row["Weight"] / (row["Height"] ** 2)
    return bmi


def add_bmr(row: pd.Series) -> float:
    bmr = (10 * row["Weight"]) + (625 * row["Height"]) - (5 * row["Age"])
    bmr += 5 if row["Gender"] == "male" else -161
    return bmr


if __name__ == "__main__":
    df = pd.read_csv("data/Obesity Dataset - Cleaned.csv")
    df["BMI"] = df.apply(add_bmi, axis=1)
    df["BMR"] = df.apply(add_bmr, axis=1)
    print(df.head())
    df.to_csv("data/Obesity Dataset - Engineered.csv", index=False)
