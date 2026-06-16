import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Load dataset
data = pd.read_csv("employee_salary.csv")

print("Dataset Preview")
print(data.head())

# Features
X = data[["Experience", "Age"]]

# Target
y = data["Salary_High"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=1
)

# Model
model = DecisionTreeClassifier()

# Training
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(acc * 100, 2), "%")

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix")
print(cm)

# Visualization
display = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

display.plot()

plt.title("Employee Salary Prediction")

plt.savefig("confusion_matrix.png")

plt.show()