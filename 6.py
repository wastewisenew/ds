from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
iris = load_iris()
x = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression(C=1e4))
model.fit(X_train, y_train)
training_accuracy = model.score(X_train, y_train)
print(f"Training Accuracy: {training_accuracy}")
testing_accuracy = model.score(X_test, y_test)
print(f"Testing Accuracy: {testing_accuracy}")