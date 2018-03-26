#!/usr/bin/python

import sys
import pickle
import math
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#features_list = ['poi','salary','total_payments','from_poi_to_this_person','from_this_person_to_poi','exercised_stock_options','shared_receipt_with_poi','fraction_from_poi','fraction_to_poi','fraction_exercised']
#features_list = ['poi','total_payments','from_poi_to_this_person','from_this_person_to_poi','exercised_stock_options','shared_receipt_with_poi','fraction_from_poi','fraction_to_poi','fraction_exercised']
#features_list = ['poi','total_payments','from_this_person_to_poi','exercised_stock_options','shared_receipt_with_poi','fraction_to_poi']
#features_list = ['poi','exercised_stock_options','shared_receipt_with_poi','fraction_to_poi']
#features_list = ['poi','exercised_stock_options','shared_receipt_with_poi','fraction_exercised']
features_list = ['poi','total_payments','exercised_stock_options','shared_receipt_with_poi','fraction_exercised']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "rb") )

### Task 2: Remove outliers
##From the plot done in lesson 7, see that we need to remove the "TOTAL" key that is part of the insiderpay.pdf
data_dict.pop("TOTAL")

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

def computeFraction( poi_messages, all_messages ):

    fraction = 0.
    if (math.isnan(float(poi_messages)) | math.isnan(float(all_messages)) ):
        return fraction
    fraction = float(poi_messages)/float(all_messages)
    return fraction


poi_nan_count=0
non_poi_nan_count=0
poi_count=0
non_poi_count=0

for key in data_dict.keys():
    
    data_point = data_dict[key]

    #new feature "fraction from poi" done in lesson 11
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi

    #new feature "fraction to poi" done in lesson 11
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_point["fraction_to_poi"] = fraction_to_poi

    #created new feature fraction of stock options that were exercised
    exercised_stock_options = data_point["exercised_stock_options"]
    total_stock_value = data_point["total_stock_value"]
    fraction_exercised = computeFraction( exercised_stock_options, total_stock_value )
    data_point["fraction_exercised"] = fraction_exercised

    #if poi, do my features have NaN?
    if data_point['poi']:
       poi_count +=1
      # if (math.isnan(float(data_point['total_payments']))):
      # if (math.isnan(float(data_point['exercised_stock_options']))):
       if (math.isnan(float(data_point['shared_receipt_with_poi']))):
          poi_nan_count +=1
    else:
       non_poi_count +=1
      # if (math.isnan(float(data_point['total_payments']))):
      # if (math.isnan(float(data_point['exercised_stock_options']))):
       if (math.isnan(float(data_point['shared_receipt_with_poi']))):
          non_poi_nan_count +=1

    data_dict[key] = data_point

#debugging info
#print (poi_count)
#print (non_poi_count)
#print (poi_nan_count)
#print (non_poi_nan_count)
#for key in data_dict.keys():
#    print (data_dict[key]['fraction_to_poi'])
#    print (data_dict[key]['fraction_from_poi'])
#    print (data_dict[key]['fraction_exercised'])

my_dataset = data_dict

### Extract features and labels from dataset for local testing
#replaces NaN with 0


data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
print ("using SelectKBest for feature selection")
features = SelectKBest(f_classif, k=2).fit_transform(features, labels)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.


from sklearn import tree
from sklearn.grid_search import GridSearchCV
#best so far
print ("using DecisionTree")
clf = tree.DecisionTreeClassifier()
#clf = tree.DecisionTreeClassifier(min_samples_split=40)
#clf = tree.DecisionTreeClassifier(criterion='entropy')
#param_grid = {
#         'min_samples_split': [4,10,20,40] ,
#          }
#clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid)


#SVM
#from sklearn.svm import SVC
#from sklearn.grid_search import GridSearchCV

#param_grid = {
#         'C': [.00001, .0001, .001,1,10] ,
#          'gamma': [0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10, 100],
#         'class_weight': [None,'auto',{1:100}, {1:10}, {1:5}],
#          }
#print ("using SVC")
#clf = GridSearchCV(SVC(kernel='rbf'), param_grid)
#clf = SVC(kernel='rbf',C=1e-05,gamma=0.0001, class_weight={1:100})

#just wanted to try this one for fun
#from sklearn.ensemble import RandomForestClassifier
#print ("using RandomForest")
#clf = RandomForestClassifier()
#clf = RandomForestClassifier(class_weight={1:100})

from sklearn import cross_validation
features_train,features_test,labels_train,labels_test = cross_validation.train_test_split(features,labels,test_size=0.3,random_state=42)

#Using SelectKBest instead
#from sklearn.feature_selection import VarianceThreshold
#print ("using Variance Threshold for feature selection")
#sel = VarianceThreshold()
#features_train = sel.fit_transform(features_train)
#features_test= sel.fit_transform(features_test)


#Using SelectKBest instead
#from sklearn.feature_selection import SelectPercentile, f_classif
#print ("using SelectPercentile for feature selection")
#selector = SelectPercentile(f_classif, percentile=1)
#selector.fit(features_train, labels_train)
#features_train = selector.transform(features_train)
#features_test  = selector.transform(features_test)

#to try and understand SVM, what did predicted labels look like?
clf=clf.fit(features_train, labels_train)
#print("Best parameters set found on train set:")
#print()
#print(clf.best_estimator_)
#print("Feature Importances")
#print (clf.feature_importances_)
#accuracy
acc = clf.score(features_test, labels_test)
print (acc)
predicted_labels=clf.predict(features_test)

#precision, recall
from sklearn.metrics import precision_score
ps = precision_score(labels_test, predicted_labels)
print (ps)

from sklearn.metrics import recall_score
rs = recall_score(labels_test, predicted_labels)
print (rs)
#print (predicted_labels)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)
