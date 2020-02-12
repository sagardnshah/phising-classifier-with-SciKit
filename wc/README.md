wc.csv contains websites’ javascript source code metrics such as number of on_clicks\ counts, avg_external script_block counts and so on to train models which will classify sites as phishing or non-phishing.

Models trained: Random forests, K-nearest neighbors, logistic regression, decision trees.

Random Forests is the best model based on multiple metrics such as confusion matrix, ROC curve, AUC, etc.

each .py file will output it's respective model's predictions to wc-phish.csv, and wc-nonphish.csv
 

