import numpy
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

german_dataset_numeric = pd.read_csv("german_dataset_numeric.csv")
X = german_dataset_numeric.iloc[:, 0:23]
first_y = german_dataset_numeric.iloc[:, 24:25].values
y = []
for correcter in first_y:
    y.append(correcter[0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

algorithm = GaussianNB(priors=None, var_smoothing=1e-9)
algorithm.fit(X_train, y_train)

seconder_y_test_in_array = numpy.array(y_test)

print("Accuracy of the algorithm:", accuracy_score(seconder_y_test_in_array, algorithm.predict(X_test)))

print(y_train)
print(y_test)