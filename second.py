import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm

labeled_images = pd.read_csv('all/train.csv')
images = labeled_images.iloc[0:,1:]
labels = labeled_images.iloc[0:,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

test_images[test_images>0]=1
train_images[train_images>0]=1

clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
print(clf.score(test_images,test_labels))

test_data=pd.read_csv('all/test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data)

df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)