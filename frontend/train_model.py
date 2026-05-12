import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv('titanic (2).csv')

# Simple preprocessing: Selecting key features
# We'll fill missing age values with the median
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
X = df[features]
y = df['Survived']

# Train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Save the model to a file
joblib.dump(model, 'titanic_model.pkl')
print("Model saved as titanic_model.pkl")