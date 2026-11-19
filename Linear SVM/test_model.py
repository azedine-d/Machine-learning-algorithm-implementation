import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from svm_linear_impl import MySVMLinear

dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=10
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MySVMLinear(X_train, y_train)
model.fit()

# التنبؤ على عيّنة واحدة من الاختبار
print(model.predict(X_test[2]), "the true one:", y_test[2])