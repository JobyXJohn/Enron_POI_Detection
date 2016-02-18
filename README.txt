Best performing classifier was a DecisionTree. However SVM also performed well. Two versions of poi_id.py are presented to evaluate each. 

### DecisionTree code: ###########
 DEcisionTree was found to be the most successful algorithm in terms of precision and recall.
 Run 'poi_id_Final_DecisionTree.py'.
It uses feature_importance to rate the importance of features. 
The most important features are selected manually. After three iterations, the best performance is obtained by retaining the top 8 features.
##############################################
 
#### SVM Code #####
poi_id_SVM.py  is identical to the above code (poi_id_Final_DecisionTree.py) except the portions relevant to SVM implementation have been uncommented.
Running it allows the GridSearchCV to run and choose optimal parameters for the SVM.
The features are selected using SelectKBest and comparing performance. It is found that having 7 best features gives the best performance.
###################

####### Plot code ###
'plot_features.py' is called to plot figures from within 'poi_id_*.py'.
 Useful in identifying outlier.
A few lines of commented code at the end of this file helps to identify outliers.
################################

##### tester.py ############
Even though CrossValidation code using StratifiedKFold is include in 'poi_id_*.py', 
the output of 'tester.py' is relied upon to evaluate the performance of the classifiers.
#################################################


