import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# 1. Create a small simulated dataset
data = pd.DataFrame({
    "age":        [22, 25, 47, 52, 46, 56, 23, 27, 48, 51],
    "income":     [25, 30, 80, 90, 75, 95, 28, 32, 85, 88],
    "past_spend": [50, 60, 300, 350, 280, 400, 55, 65, 320, 340],
    "purchase":   [0,  0,  1,   1,   1,   1,   0,  0,  1,   1]
})

# 2. Train the model
X = data[["age", "income", "past_spend"]]
y = data["purchase"]

model = LogisticRegression()
model.fit(X, y)

# 3. Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")