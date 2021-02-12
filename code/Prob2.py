import pandas as pd
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
glass_data = pd.read_csv('glass.csv')
x=glass_data.drop('Type',axis=1)
y=glass_data['Type']
X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3,random_state=0)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("classification_report\n",metrics.classification_report(y_test,y_pred))