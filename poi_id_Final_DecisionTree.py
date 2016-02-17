#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
sys.path.append("../final_project/")
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from plot_features import *
import warnings
warnings.filterwarnings('ignore')

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

full_feature = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 
'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 
'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
 'director_fees','to_messages','from_poi_to_this_person', 'from_messages',
 'from_this_person_to_poi', 'shared_receipt_with_poi','Topoi2frm','frmpoi2To',
 'poi_mail_ratio','bon2sal','stock2sal','is_enron_emp']
#choice_index = [0,3,10,12,13,16,19,20,22] # always choose 0 for 'poi'
## use full feature list and depend on selectKbest for SVM
choice_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

## For a DECISIONTREE, use feature_importance to find best features.

#choice_index  =[0,1,5 ,8 , 9 ,10,12 ,18 ,19, 20 ,23]
features_list = list( full_feature[i] for i in choice_index )
print features_list
####### GOOD FEATURE LIST

## FULL FEATURE LIST USED FOR TESTING

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL',0) # too high 'bonus'
data_dict.pop('LAY KENNETH L',0) # too high 'bonus'
data_dict.pop('SKILLING JEFFREY K',0) # too high 'bonus'
#data_dict.pop('LAVORATO JOHN J',0) # too high 'total_payments'
#data_dict.pop('BHATNAGAR SANJAY',0) # too high 'total_payments'
#data_dict.pop('FREVERT MARK A',0) # too high bonus

### Task 3: Create new feature(s)
for item in data_dict:
#    print type(data_dict[item]['salary']),data_dict[item]['salary'],data_dict[item]['salary']=='NaN'
    # Ratio of Bonus to Salary
    if data_dict[item]['salary']!='NaN' and data_dict[item]['bonus']!='NaN':
        value = float(data_dict[item]['bonus'])/float(data_dict[item]['salary'])
        #print(value)
    else:
        value='NaN'
    data_dict[item].update({'bon2sal':value})
    
    #Ratio of Total_stock_value to salary
    if data_dict[item]['salary']!='NaN' and data_dict[item]['total_stock_value']!='NaN':
        value = float(data_dict[item]['total_stock_value'])/float(data_dict[item]['salary'])
        #if value >1:        
           # print value, data_dict[item]['poi']
    else:
        value='NaN'
    data_dict[item].update({'stock2sal':value})
    
    # Ratio of mails to poi to total mails sent
    # Ratio of mails from poi to total mails received.
    # and Ration of total mails (to or from) poi to total mails sent or received.
    if data_dict[item]['from_this_person_to_poi']!='NaN' and data_dict[item]['from_messages']!='NaN':
        value= float(data_dict[item]['from_this_person_to_poi'])/float(data_dict[item]['from_messages'])
        valu2= float(data_dict[item]['from_poi_to_this_person'])/float(data_dict[item]['to_messages'])
        valu3= float((data_dict[item]['from_this_person_to_poi']+data_dict[item]['from_poi_to_this_person']))/\
        float(data_dict[item]['to_messages']+data_dict[item]['from_messages'])
    else:
        value,valu2 = 'NaN','NaN'
    data_dict[item].update({'Topoi2frm':value,'frmpoi2To':valu2,'poi_mail_ratio':valu3})
    
    # Is person an enron employee
    if data_dict[item]['email_address']=='NaN':
        value=0
    else:
        value =1
    data_dict[item].update({'is_enron_emp':value})    
    
### Store to my_dataset for easy export below.
### Extract features and labels from dataset for local testing
my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys = True)

##################
## Rescale the features:
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
Scaled_data = min_max_scaler.fit_transform(data)
labels, features = targetFeatureSplit(Scaled_data) # Full feature set of 25 features
##### The following function plots 2 figures. a 2d plot and a 3d plot. 
#plot_features(data,data_dict,features_list,[1,10],[1,9,10])

# Numbers above (2 lists) indicate zero based indexing of features_list

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB() # CLASSIFIER 1


## %%%%%%%%%% The following classifier meets the critertion of precision and recall>0.3

#### ########### FEATURE SELECTION ###########################################

#from sklearn import feature_selection
#### %%%%%%%%%% FOR SVM ############
#ch2 = feature_selection.SelectKBest(feature_selection.chi2,k=25) # keep 7 for 
###                                                                  best svm
#features_train = ch2.fit_transform(features, labels)
#choice_index = ch2.get_support(indices=True)
#print np.vstack((ch2.scores_,ch2.pvalues_))
#new_selected_features = list( features_list[i+1] for i in choice_index )
#print ' '
#print 'The selected features are:', new_selected_features
#print 'And their indices are:', choice_index+1
#print ' '
#
#new_features_list = ['poi']+new_selected_features
#data = featureFormat(my_dataset, new_features_list, sort_keys = True)
### Rescale the features:
#Scaled_data = min_max_scaler.fit_transform(data)
#labels, features = targetFeatureSplit(Scaled_data)
#### %%%%%%%%%%%%%%%%%%% ###########

########## FOR DECISION TREE ##################################################
####### Use FEATURE_IMPORTANCE to choose most important features
### Comment this section while running svm ##########
choice_index = [8,9,10,11,13,19,20,22] ## Best set of features for Decision Trees.
# The above list is found through **three iterations** of feature_importance
# and eliminating the least important features. Each time the performance as 
# reported by tester.py is checked. The best performance is given by the first
# list
new_features_list = list( full_feature[i] for i in choice_index )
new_features_list = ['poi']+new_features_list

print new_features_list
data = featureFormat(my_dataset, new_features_list, sort_keys = True)
## Rescale the features:
Scaled_data = min_max_scaler.fit_transform(data) # Reduced data
labels, features = targetFeatureSplit(Scaled_data)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0) #CLASSIFIER 2
clf.fit(features,labels)
print clf.feature_importances_ 

## %%%%%%%% ADA BOOST DECISION TREE %%%%%%%
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.tree import DecisionTreeClassifier
#clf = AdaBoostClassifier(DecisionTreeClassifier(random_state=10),
#                        algorithm="SAMME",
#                         n_estimators=200)

##%%%%%%% SUPPORT VECTOR CLASSIFIER %%%%%%%%%%%
#from sklearn import svm
##### TUNING PARAMETERS ###### 
#from sklearn.grid_search import GridSearchCV
#svr = svm.SVC()
#parameters = {'C':[1e3,1e4,5e4,1e5,5e5,1e6],
#              'gamma': [0.001, 0.005, 0.01,0.05,0.1]}
##
#scores = ['f1'] 
#for score in scores: # could have other measures also.
#    print("# Tuning hyper-parameters for %s" % score)
#    clf_grid = GridSearchCV(svr, parameters,scoring='%s' % score,cv=5)
#    clf_grid.fit(features, labels)
#
#    print("Best parameters set found on development set:")
#    print()
#    print(clf_grid.best_params_)
#    print()
#clf = clf_grid.best_estimator_ #### PASS THIS CLASSIFIER TO TESTER.PY
#clf =svm.SVC(C=5e4,gamma=0.1)
#######################################################

###########
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import StratifiedKFold
skf =  StratifiedKFold(labels, n_folds=3, shuffle=True,random_state=0)
train_idx, test_idx = next(iter(skf))
features_train  = list(features[ii] for ii in train_idx)
labels_train = list(labels[ii] for ii in train_idx)
features_test = list(features[ii] for ii in test_idx)
labels_test = list(labels[ii] for ii in test_idx)
# USE the CHOSEN clf to fit and predict 
clf.fit(features_train,labels_train)
labels_pred = clf.predict(features_test) # predict the labels on test features

#### Evaluate Precision and Recall
from sklearn.metrics import precision_score, recall_score
prec = precision_score(labels_test,labels_pred) # Precision
reca = recall_score(np.array(labels_test),labels_pred)    # Recall
scr = clf.score(features_test,labels_test)      # Accuracy
print
#print 'Precison, REcall, Score, are:', prec,'and', reca, 'and', scr
print

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, new_features_list,min_max_scaler)
execfile('tester.py') # So that we don't have to run two files to get the
                    #precision and recall