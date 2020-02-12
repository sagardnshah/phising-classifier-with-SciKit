cf.csv contains certificate data for 8399 websites which will be used to train various models to classify sites as phishing or non-phishing.

Models trained: Random forests, K-nearest neighbors, logistic regression, decision trees. 

Random Forests is the best model based on multiple metrics such as confusion matrix, ROC curve, AUC, etc.

each .py file will output it's respective model's predictions to cf-phish.csv, and cf-non-phish.csv
