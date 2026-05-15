import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
import joblib

# 1. Load Data
df = pd.read_csv('online_vs_offline_learning_dataset.csv')

# 2. Preprocessing
le = LabelEncoder()
df['Learning_Mode_Encoded'] = le.fit_transform(df['Learning_Mode'])
# Subjects-ah numbers-ah mathurathukku
df = pd.get_dummies(df, columns=['Subject'], drop_first=True)

# Features (X) and Target (y)
X = df.drop(['Learning_Mode', 'Exam_Score'], axis=1)
y = df['Exam_Score']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 3. Model Training (Linear Regression)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save Models
joblib.dump(model, 'score_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'features.pkl')

print("Success! Models and Scaler saved.")