import pandas as pd
from sklearn import model_selection
from sklearn import svm
from sklearn import metrics
glass_data = pd.read_csv('glass.csv')
x=glass_data.drop('Type',axis=1)
y=glass_data['Type']
X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3,random_state=0)
lc = svm.SVC(kernel="linear")
lc.fit(X_train, y_train)
y_pred = lc.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("classification_report\n",metrics.classification_report(y_test,y_pred,zero_division=1))
#print("confusion matrix\n",metrics.confusion_matrix(y_test,y_pred))