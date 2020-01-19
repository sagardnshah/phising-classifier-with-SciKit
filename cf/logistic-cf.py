#importing libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

#importing the dataset
df = pd.read_csv("cf.csv")

dataset = df.drop(columns=['domain']) #removing unwanted column

dataset[['has_cert', 'valid_cert', 'extended_validation','multi_mtn','class']]*= 1

dataset['class'].replace({"alexa":0}, inplace=True)
dataset['class'].replace({"phish":1}, inplace=True)



x = dataset.iloc[ : , :-1].values
y = dataset.iloc[:, -1:].values

#spliting the dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.30, random_state =2 )

#fitting logistic regression 
classifier = LogisticRegression(random_state = 2)
classifier.fit(x_train, y_train)

#predicting the tests set result
y_pred = classifier.predict(x_test)

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#classification report
print(classification_report(y_test, y_pred))

#accuracy in percantage
accuracy=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print(accuracy)

y_pred_prob = classifier.predict_proba(x_test)[:,1]
fpr,tpr,thresh = roc_curve(y_test,y_pred_prob) 
roc = accuracy_score(y_test,y_pred)

print('AUC: '+ str(roc_auc_score(y_test, y_pred_prob)))

# Plot ROC curve for Logistic Regression
plt.plot(fpr,tpr,'orange',label = 'Logistic Regression')
plt.legend("Logistic Regression", loc='lower right')
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend(loc='lower right')
