import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

# 1. Load the Iris Dataset (Direct-ah sklearn-laye irukum)
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 2. Training and Testing split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. SVM Model Training (Linear Kernel use panrom)
model = SVC(kernel='linear') 
model.fit(X_train, y_train)

# 4. Model-ah file-ah save panrom
with open('iris_svm_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("SVM Model trained and saved as 'iris_svm_model.pkl'!")