import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

iris = pd.read_csv("C:/Users/ACER/IRIS.csv")

print(iris.head())
print(iris.describe())
print(iris.info())
print(iris[iris.isnull()].count())

print(iris['species'].unique())
print(iris['species'].value_counts())

iris.plot(kind="scatter",x="sepal_length",y="sepal_width")
plt.show()
iris.plot(kind="scatter",x="petal_length",y="petal_width")
plt.show()
iris.hist(bins=50,figsize=(20,15))
plt.show()
sns.pairplot(iris)
plt.show()
sns.heatmap(iris.corr(),annot=True)
plt.show()
iris.plot(kind="bar",x="sepal_length")
plt.show()
iris.plot(kind="bar",x="petal_length")
plt.show()
sns.boxplot("petal_length","sepal_width",data=iris)
plt.show()
iris["sepal_length"].value_counts().plot(kind="bar")
iris["petal_width"].value_counts().plot(kind="bar")
iris["sepal_length"].value_counts().plot(kind="bar")
iris["sepal_width"].value_counts().plot(kind="bar")
plt.show()


from sklearn.model_selection import train_test_split
X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=101)

#Training a Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
tr=DecisionTreeClassifier()
tr.fit(X_train,y_train)
pred=tr.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

#Training a RandomForest Model
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(n_estimators=600)
forest.fit(X_train,y_train)
pre=forest.predict(X_test)
print(confusion_matrix(y_test,pre))
print(classification_report(y_test,pre))

#Training a LogisticRegressio Model
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

predlog = logmodel.predict(X_test)
print(confusion_matrix(y_test,predlog))
print(classification_report(y_test,predlog))

#Training svm model
from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
predsvm = clf.predict(X_test)
print(confusion_matrix(y_test,predsvm))
print(classification_report(y_test,predsvm))

#naive_bayes model
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
predn = gnb.predict(X_test)
print(confusion_matrix(y_test,predn))
print(classification_report(y_test,predn))