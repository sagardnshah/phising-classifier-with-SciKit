import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score


#importing the dataset
df = pd.read_csv("wc.csv")

dataset=df


dataset['type'].replace({"alexa":0}, inplace=True)
dataset['type'].replace({"phish":1}, inplace=True)

dataset[['has_html']]*= 1

x = dataset.drop(['type'],axis=1)
y = dataset['type'].values

#spliting the dataset into training set and test set

x_train1, x_test1, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state =0 )

x_train=x_train1.drop(columns=['site','scrap_time','path'])
x_test=x_test1.drop(columns=['site','scrap_time','path'])

#----------------applying grid search to find best performing parameters 
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [100, 500],
    'max_features': ['sqrt', 'log2'],
    'criterion' :['gini', 'entropy']}]

grid_search = GridSearchCV(RandomForestClassifier(),  parameters,cv =5, n_jobs= -1)
grid_search.fit(x_train, y_train)
#printing best parameters 
print("Best Accurancy =" +str( grid_search.best_score_))
print("best parameters =" + str(grid_search.best_params_)) 
#-------------------------------------------------------------------------

#fitting RandomForest regression with best params 
classifier = RandomForestClassifier(n_estimators = 100, criterion = "gini", max_features = 'log2',  random_state = 0)
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


#roc
y_pred_prob = classifier.predict_proba(x_test)[:,1]
fpr,tpr,thresh = roc_curve(y_test,y_pred_prob) 
roc = accuracy_score(y_test,y_pred)

print('AUC: '+ str(roc_auc_score(y_test, y_pred_prob)))


# Plot ROC curve for Logistic Regression
plt.plot(fpr,tpr,'orange',label = 'Random Forest')
plt.legend("Logistic Regression", loc='lower right')
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend(loc='lower right')

#copy predicted result to a dataframe
df2= pd.DataFrame()
df2 = pd.DataFrame(y_pred)
Predicted_Result=df2.rename(columns ={0: "Predicted Results"})

#copy real results to dataframe
Actual_Result= pd.DataFrame() 
Actual_Result = pd.DataFrame(y_test)
Actual_Result=Actual_Result.rename(columns ={0: "Actual Results"})

#concate testing data + predicted result + real results
df3=x_test1
df3['Predicted Results'] = Predicted_Result['Predicted Results'].values
df3['Actual Results'] = Actual_Result['Actual Results'].values

#final testing data 
final_df=df3

#change 1 back to phish and 0 to Alexa
final_df['Predicted Results'].replace({0:"alexa"}, inplace=True)
final_df['Predicted Results'].replace({1:"phish"}, inplace=True)

final_df['Actual Results'].replace({0:"alexa"}, inplace=True)
final_df['Actual Results'].replace({1:"phish"}, inplace=True)

#get all predicted phishing sites
phish=final_df.loc[final_df['Predicted Results'] == 'phish']

#get all predicted non phishing sites
nonphish=final_df.loc[final_df['Predicted Results'] == 'alexa']

phish.to_csv (r"C:\Users\sshah65\Google Drive\UNCC\Security Analytics\Project 3 - Phising Classifier with SciKit\code\wc\wc-phish.csv", index = None, header=True)
nonphish.to_csv (r"C:\Users\sshah65\Google Drive\UNCC\Security Analytics\Project 3 - Phising Classifier with SciKit\code\wc\wc-nonphish.csv", index = None, header=True)
