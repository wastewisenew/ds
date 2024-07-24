import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
kernels = ['rbf']
gammas = [0.5]
Cs = [0.01, 1, 10]
best_accuracy = 0
best_support_vectors = None
for kernel in kernels:
    for gamma in gammas:
        for C in Cs:
            model = SVC(kernel=kernel, gamma=gamma, C=C, decision_function_shape='ovr')
            model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
total_support_vectors = np.sum(model.n_support_)
if accuracy > best_accuracy:
    best_accuracy = accuracy
    best_support_vectors = total_support_vectors
print(f"Kernel: {kernel}, Gamma: {gamma}, C: {C}, Accuracy: {accuracy}, Total Support
Vectors: {total_support_vectors}")