import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 1. Load the dataset
columns = ["Year", "Month", "Day", "Decimal Date", "Sunspot Number", "Standard Deviation", "Number of Observations", "Definitive/Provisional Indicator"]
data = pd.read_csv("sunspot.csv", sep=";", names = columns, skiprows = 1) # https://www.sidc.be/SILSO/home
print(data.head())
sunspots = data["Sunspot Number"].values

# 2. mormalize the data
scaler = MinMaxScaler(feature_range = (0,1))
sunspots = scaler.fit_transform(sunspots.reshape(-1, 1)).flatten()

# 3. prepare the dataset
def create_dataset(data, n_steps):
    x, y = [], []
    for i in range(len(data) - n_steps):
        x.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(x), np.array(y)

n_steps = 10 # can adjust
x, y = create_dataset(sunspots, n_steps)
x = x.reshape(-1, n_steps, 1)

# 4. split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# 5. create the model
def create_RNN(hidden_units = 50, seq_length = 3):
    model = Sequential([
        SimpleRNN(hidden_units, input_shape = (seq_length, 1)),
        Dense(1)
    ])
    model.compile(loss = "mean_squared_error", optimizer = "adam")
    return model

# n_steps = 3
# x, y = [], []
# for i in range(len(data) - n_steps):
#     x.append(data[i:i + n_steps])
#     y.append(data[i + n_steps])
# x, y = np.array(x), np.array(y)
# x = x.reshape(-1, n_steps, 1)


# Create the model
model = create_RNN()
model.fit(x, y, epochs = 100, batch_size = 16, validation_data = (x_test, y_test))

# Save the model
model.save("sunspot_RNN.h5")