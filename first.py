import numpy as np
import matplotlib.pyplot as  plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("all/train.csv").as_matrix()
clf = DecisionTreeClassifier()

#training dataset
xtrain = data[0:21000,1:]
train_label = data[0:21000,0]

clf.fit(xtrain,train_label)

#testing data
xtest = data[21000:,1:]
actual_label = data[21000:,0]

p = clf.predict(xtest)

print(clf.score(xtest,actual_label))