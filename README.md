# Obesity Level Prediction

A supervised machine learning project developed as part of **Konecta’s AI/ML Internship Program** to predict a patient's
**obesity level** based on lifestyle habits. The project features a complete ML pipeline — from **data cleaning** and
**exploratory data analysis** to **model training**, **evaluation**, and **deployment** with a simple **Flask GUI**.

🏆 **Awarded 1st place** in Konecta's Internship Cycle 1 graduation projects.

![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-orange.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-informational)

---

## 📂 Table of Contents

- [Dataset Credits](#-dataset-credits)
- [Project Features](#-project-features)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

---

## 🎓 Dataset Credits

This project utilizes the **Obesity Prediction Dataset**, created and shared
by [Stephen Adeniran on Kaggle](https://www.kaggle.com/datasets/adeniranstephen/obesity-prediction-dataset).

> Please refer to the dataset page for further details, licensing, and terms of use.

### 🧾 Dataset Features

| Feature Name              | Description                               |
|---------------------------|-------------------------------------------|
| Gender                    | Male or Female                            |
| Age                       | Age in years                              |
| Height                    | Height in meters                          |
| Weight                    | Weight in kilograms                       |
| Overweight_Family_History | Family history with overweight (Yes/No)   |
| High_Calorie_Consumption  | Frequent consumption of high-calorie food |
| Vegetable_Consumption     | Frequency (1 to 3)                        |
| Main_Meals                | Number of main meals per day              |
| Snack_Consumption         | Consumption between meals                 |
| Smoker                    | Whether the person smokes                 |
| Water_Intake              | Daily intake (1 to 3)                     |
| Calorie_Monitoring        | Whether calories are monitored            |
| Physical_Activity         | Frequency of activity (0 to 3)            |
| Tech_Time                 | Screen time (0 to 2)                      |
| Alcohol_Consumption       | Alcohol frequency                         |
| Transportation_Mean       | Main means of transport                   |
| Obesity_Level (Target)    | Obesity classification                    |
| BMI                       | Body Mass Index                           |
| BMR                       | Basal Metabolic Rate                      |

---

## 🌟 Project Features

- Full ML pipeline:
    - Data cleaning
    - Exploratory data analysis
    - Feature engineering
    - Data preprocessing
    - Model training & evaluation
    - Model comparison
    - Model interface creation
    - Flask-based GUI
- Modular design with `Model_Interface.py` for easy integration

---

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Flask
- Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- Trained with multiple classifiers (Gradient Boosting, SVM, Logistic Regression, etc.)

---

## ⚙️ Installation

```bash
git clone https://github.com/Ahmed-Madeh-Noah/Obesity-Level-Prediction.git
cd Obesity-Level-Prediction
pip install -r requirements.txt
```

---

## 🧪 Usage

### 🔘 Option 1: With Flask GUI

```bash
python app.py
```

- Open `http://127.0.0.1:8000/` in your browser
- Fill out the form and receive a prediction instantly

### 🔧 Option 2: Programmatically (Without GUI)

Import and call the model interface directly:

```python
from Model_Interface import predict_from_input
import pandas as pd

user_input = pd.Series({
    'Age': 23,
    'Height': 1.75,
    # ...
})

prediction = predict_from_input(user_input)
print(prediction)
```

> ⚠️ Ensure your `pd.Series` matches the expected feature names and constraints exactly — no validation is done by the
> function.

---

## 📊 Model Performance

The following table summarizes the accuracy and other metrics for several classifiers:

| Model                | Train Acc | Val Acc   | Test Acc  | Balanced Acc | Precision | F1 Score  |
|----------------------|-----------|-----------|-----------|--------------|-----------|-----------|
| GaussianNB           | 0.822     | 0.794     | 0.811     | 0.810        | 0.807     | 0.805     |
| LogisticRegression   | 0.962     | 0.950     | 0.957     | 0.956        | 0.958     | 0.957     |
| KNeighbors           | 0.930     | 0.892     | 0.847     | 0.841        | 0.849     | 0.838     |
| DecisionTree         | 1.000     | 0.976     | 0.978     | 0.977        | 0.979     | 0.978     |
| SVC                  | 0.972     | 0.955     | 0.962     | 0.961        | 0.962     | 0.962     |
| RandomForest         | 0.962     | 0.955     | 0.947     | 0.945        | 0.948     | 0.947     |
| **GradientBoosting** | **0.966** | **0.962** | **0.959** | **0.959**    | **0.961** | **0.959** |
| MLP                  | 0.965     | 0.914     | 0.938     | 0.936        | 0.938     | 0.938     |

📈 The best-performing model was **Gradient Boosting Classifier**.

---

## 👥 Acknowledgements

This project was created as part of the **Konecta AI/ML Internship Program**.

- **Instructor**: Mostafa Atlam
- **Team Members**:
  Ahmed Noah, Dina Hossam, Hala Hussein, Mariam Amr, Marwan Saad, Menna Mohamed, Noreen Amir

---

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## 💬 Feedback & Discussions

Although this project is finalized and submitted, we welcome **discussions, questions, and suggestions**. Feel free to
open an issue or leave a comment!
