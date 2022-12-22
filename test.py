import pandas as pd
import numpy as np
iris = pd.read_csv("IRIS.csv")
# splitting the dataset
x = iris.drop("species", axis=1)
y = iris["species"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2,
                                                    random_state=0)
# training the model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
# giving inputs to the machine learning model
# features = [[sepal_length, sepal_width, petal_length, petal_width]]
features = np.array([[5, 2.9, 1, 0.2]])
# using inputs to predict the output
prediction = knn.predict(features)
print("Prediction: {}".format(prediction))