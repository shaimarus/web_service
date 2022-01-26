import numpy as np
from sklearn import datasets
import pickle
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

iris_X, iris_y = datasets.load_iris(return_X_y=True)

knn = KNeighborsClassifier()
model_xgb=xgb.XGBClassifier()

model_xgb.fit(iris_X, iris_y)
print(iris_X[0])
pickle.dump(model_xgb,open('model.pkl', 'wb'))
