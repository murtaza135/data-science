import numpy as np
import pandas as pd
import joblib

data = pd.read_csv("test.csv")
data = np.array(data.iloc[[0]])

model = joblib.load("models/model.joblib")
prediction = model.predict(data)
print(f"Group: {prediction[0]}")