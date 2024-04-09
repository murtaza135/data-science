import numpy as np
import pandas as pd
import joblib

diabetes_mapper = {
    0: "It is NOT likely",
    1: "It is likely"
}

data = pd.read_csv("test.csv")
data = np.array(data[["Glucose", "BMI"]].iloc[[0]])

svm = joblib.load("models/svm.joblib")
prediction = svm.predict(data)
print(f"Will you get diabetes? {diabetes_mapper[prediction[0]]}")