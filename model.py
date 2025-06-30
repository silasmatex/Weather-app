import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("data/cameroon_weather.csv")

# Encode target variable
df['Condition'] = df['Condition'].astype('category')
df['Condition_code'] = df['Condition'].cat.codes

# Features and target
X = df[['Temp', 'Humidity', 'Rainfall', 'WindSpeed']]
y = df['Condition_code']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/weather_model.pkl")
