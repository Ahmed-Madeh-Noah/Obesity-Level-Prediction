"""
Model Evaluation Script
This script defines the class needed to evaluate different classification models.
If the script is run directly, it initializes an empty evaluations dataframe.

Example Usage:
from Model_Evaluation import ModelsEvaluator

models_evaluator = ModelsEvaluator(X_train, y_train, X_val, y_val, X_test, y_test)
models_evaluator.evaluate("GaussianNB", GaussianNB(), save_model=True)
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt


class ModelsEvaluator:
    def __init__(self, xtrain: pd.DataFrame, ytrain: np.ndarray, xval: pd.DataFrame, yval: np.ndarray,
                 xtest: pd.DataFrame, ytest: np.ndarray):
        self.evaluations = pd.DataFrame(
            columns=["Train_Accuracy", "Val_Accuracy", "Test_Accuracy", "Balanced_Accuracy", "Precision", "Recall",
                     "F1_Score"])
        self.evaluations.index.name = "Model"
        self.X_train = xtrain
        self.y_train = ytrain
        self.X_val = xval
        self.y_val = yval
        self.X_test = xtest
        self.y_test = ytest

    def evaluate(self, model_name: str, model: BaseEstimator, save_model: bool = False) -> pd.Series:
        model.fit(self.X_train, self.y_train)
        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_val)
        test_pred = model.predict(self.X_test)
        self.evaluations.loc[model_name, "Train_Accuracy"] = accuracy_score(self.y_train, train_pred)
        self.evaluations.loc[model_name, "Val_Accuracy"] = accuracy_score(self.y_val, val_pred)
        self.evaluations.loc[model_name, "Test_Accuracy"] = accuracy_score(self.y_test, test_pred)
        self.evaluations.loc[model_name, "Balanced_Accuracy"] = balanced_accuracy_score(self.y_test, test_pred)
        self.evaluations.loc[model_name, "Precision"] = precision_score(self.y_test, test_pred, average="weighted",
                                                                        zero_division=0)
        self.evaluations.loc[model_name, "Recall"] = recall_score(self.y_test, test_pred, average="weighted",
                                                                  zero_division=0)
        self.evaluations.loc[model_name, "F1_Score"] = f1_score(self.y_test, test_pred, average="weighted",
                                                                zero_division=0)
        if save_model:
            joblib.dump(model, f"models/{model_name}.pkl")
        return self.evaluations.loc[model_name].copy()

    def get_all_evaluations(self) -> pd.DataFrame:
        return self.evaluations.copy()

    def line_plot(self, filename: str = None) -> None:
        df = self.get_all_evaluations()
        df['Avg'] = df.mean(axis=1)
        df = df.sort_values('Avg').drop(columns='Avg')
        df['Model'] = df.index
        df = df.melt(id_vars='Model', var_name='Metric', value_name='Score')
        sns.lineplot(data=df, x='Model', y='Score', hue='Metric', marker='o')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(f"{filename}.png")
        plt.show()


if __name__ == "__main__":
    X_train = pd.read_csv("data/preprocessed data/X_train.csv")
    X_val = pd.read_csv("data/preprocessed data/X_val.csv")
    X_test = pd.read_csv("data/preprocessed data/X_test.csv")
    y_train = pd.read_csv("data/preprocessed data/y_train.csv", header=None).to_numpy().ravel()
    y_val = pd.read_csv("data/preprocessed data/y_val.csv", header=None).to_numpy().ravel()
    y_test = pd.read_csv("data/preprocessed data/y_test.csv", header=None).to_numpy().ravel()
    models_evaluator = ModelsEvaluator(X_train, y_train, X_val, y_val, X_test, y_test)
    print(models_evaluator.get_all_evaluations())
