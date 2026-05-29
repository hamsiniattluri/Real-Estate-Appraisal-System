import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
data = pd.read_csv("house_data.csv")

# Convert location text into numbers
encoder = LabelEncoder()
data['location'] = encoder.fit_transform(data['location'])

# Inputs
X = data[['location', 'area', 'bedrooms', 'bathrooms']]

# Output
y = data['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model and encoder
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(encoder, open("encoder.pkl", "wb"))

print("Model trained successfully!")