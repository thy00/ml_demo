import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. load the dataset
# data = pd.read_csv("sunspot.csv", sep = ";", header = None)
data = pd.read_csv("EISN_current.csv", sep = ",", header = None)
columns = ["Year", "Month", "Day", "Decimal Date", "Sunspot Number", "Standard Deviation", "Number of Observations", "Definitive/Provisional Indicator"]
data.columns = columns
sunspots = data["Sunspot Number"].values

# 2. normalize the data and prepare the dataset
scaler = MinMaxScaler(feature_range=(0,1))
sunspots_scaled = scaler.fit_transform(sunspots.reshape(-1, 1)).flatten()

# 3. create the dataset
def create_dataset(data, n_steps):
    x = []
    for i in range(len(data) - n_steps):
        x.append(data[i : i +n_steps])
    return np.array(x)

n_steps = 10
x_input = create_dataset(sunspots_scaled, n_steps)

# 4. load the model
model = load_model("sunspot_RNN.h5")

# 5. make predictions
predictions_scaled = model.predict(x_input)

# 6. inverse transform the predictions
predictions = scaler.inverse_transform(predictions_scaled)

# 7. plot the results
time_series = data["Decimal Date"].values[n_steps:]

plt.figure(figsize=(12, 6))
plt.plot(data["Decimal Date"], sunspots, label = "ACtual Sunspot Numbers", color = "blue")
plt.plot(time_series, predictions, label = "Predicted Sunspot Numbers", color = "red", linestyle = "--")
plt.title("Sunspot Number Prediction")
plt.xlabel("Decimal Date")
plt.ylabel("Sunspot Number")
plt.legend()
plt.grid()
plt.show()
