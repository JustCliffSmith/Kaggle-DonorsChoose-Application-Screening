# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 15:53:04 2018

By: Justin Clifford Smith
"""

import numpy as np 
import pandas as pd
#import matplotlib.pyplot as plt
#from scipy import stats

#import re
#from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale,LabelBinarizer, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
#from sklearn.decomposition import PCA, TruncatedSVD

#%%
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
resources = pd.read_csv('./resources.csv')

# for testing purposes
#train = train.iloc[:18208,:] 
#test = test.iloc[:7803,:]

print(train.shape)
print(test.shape)
print(resources.shape)

id_cost = pd.DataFrame({'id': resources['id'], 'total_cost': resources['quantity'] * resources['price']})
id_total_cost = id_cost.groupby(id_cost['id'], sort=False).sum().reset_index()

train_resources = pd.merge(train, id_total_cost, on='id', sort=False)
test_resources = pd.merge(test, id_total_cost, on='id', sort=False)

del train
del test
del resources
del id_cost
del id_total_cost

print(train_resources.shape)
print(test_resources.shape)

#%%
print(train_resources.columns)
#%%
    
""" Performs all the preprocessing of the data."""
print('==== Beginning preprocessing of the data ====')

scaler = StandardScaler()
numeric_labels = ['teacher_number_of_previously_posted_projects', 'total_cost']
for label in numeric_labels:
    train_resources[label] = scaler.fit_transform(train_resources[label].astype(np.float64).values.reshape(-1, 1))
    test_resources[label] = scaler.transform(test_resources[label].astype(np.float64).values.reshape(-1, 1))
"""
category_labels = ['school_state', 'project_grade_category', 'project_subject_categories']
for label in category_labels:
    lb = LabelBinarizer()
    train_resources[label] = lb.fit_transform(train_resources[label])
    test_resources[label] = lb.transform(test_resources[label])
"""    
category_labels = ['school_state', 'project_grade_category']
#category_labels = ['project_grade_category']
#category_labels = []
for label in category_labels:
    lb = LabelBinarizer()
    train_resources[label] = lb.fit_transform(train_resources[label])
    test_resources[label] = lb.transform(test_resources[label]) 

train_resources['subjects'] = train_resources[['project_subject_categories', 'project_subject_subcategories']].apply(lambda x: ' '.join(x), axis=1)
test_resources['subjects'] = test_resources[['project_subject_categories', 'project_subject_subcategories']].apply(lambda x: ' '.join(x), axis=1)

#ps = PorterStemmer()
#def wordPreProcess(sentence):
#    return ' '.join([ps.stem(x.lower()) for x in re.split('\W', sentence) if len(x) >= 1])

subject_vectorizer = TfidfVectorizer(
        analyzer='word', # char ain't great
        norm='l2', # l2 is best by a lot
        token_pattern=r'\w{1,}',
        strip_accents='unicode', 
        stop_words='english',
        #preprocessor=wordPreProcess,
        max_df=1.0, # anything less than 1.0 decreased accuracy
        min_df=0, # smaller is better
        lowercase=True, # better than false
        sublinear_tf=False, # better than true
        ngram_range=(1,1), # better than (1,1)
        max_features=15000) #best is 15000 with LR
print('Fitting and transforming train subjects')
subject_vec_output_train = subject_vectorizer.fit_transform(train_resources['subjects'])
print('Transforming test subjects')
subject_vec_output_test = subject_vectorizer.transform(test_resources['subjects'])

train_resources['project_essay_3'] = train_resources['project_essay_3'].fillna('') 
train_resources['project_essay_4'] = train_resources['project_essay_4'].fillna('')
test_resources['project_essay_3'] = test_resources['project_essay_3'].fillna('') 
test_resources['project_essay_4'] = test_resources['project_essay_4'].fillna('')  

train_resources['essays'] = train_resources[['project_essay_1', 'project_essay_2','project_essay_3','project_essay_4']].apply(lambda x: ' '.join(x), axis=1)
test_resources['essays'] = test_resources[['project_essay_1', 'project_essay_2','project_essay_3','project_essay_4']].apply(lambda x: ' '.join(x), axis=1)

for label in ['project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4']:
    del train_resources[label]
    del test_resources[label]

word_vectorizer = TfidfVectorizer(
        analyzer='word', # char ain't great
        norm='l2', # l2 is best by a lot
        token_pattern=r'\w{1,}',
        strip_accents='unicode', 
        #stop_words='english', # including stop words diminished performance
        #preprocessor=wordPreProcess, # surprisingly, stemming hurt
        max_df=1.0, # anything less than 1.0 decreased accuracy
        min_df=0, # smaller is better
        lowercase=True, # better than false
        sublinear_tf=False, # better than true
        ngram_range=(1,2), # better than (1,1)
        max_features=15000) #best is 15000 with LR
print('Fitting and transforming train')
word_vec_output_train = word_vectorizer.fit_transform(train_resources['essays'])
del train_resources['essays']
print('Transforming test')
word_vec_output_test = word_vectorizer.transform(test_resources['essays'])
del test_resources['essays']

print('==== Word vectorization complete ====')

useful_data_train = []
useful_data_test = []
for label in numeric_labels + category_labels:
    useful_data_train.append(train_resources[label])
    useful_data_test.append(test_resources[label])

print('==== Forming y ====')
y = train_resources['project_is_approved'].to_sparse().as_matrix()
del train_resources

from scipy.sparse import hstack

print('==== Forming X ====')
X = pd.concat(useful_data_train, axis=1).to_sparse().as_matrix()
del useful_data_train
X = hstack((X, word_vec_output_train, subject_vec_output_train))
print('==== Forming Xtest ====')
Xtest = pd.concat(useful_data_test, axis=1).to_sparse().as_matrix()
del useful_data_test
Xtest = hstack((Xtest, word_vec_output_test, subject_vec_output_test))

print('==== Ending preprocessing of the data ====')


#%%
"""
from sklearn.model_selection import GridSearchCV
parameters = [{'penalty':['l1'],'solver':['liblinear', 'saga'], 'C':[ .1, .3, 1, 3]},
               {'penalty':['l2'],'solver':['liblinear', 'newton-cg', 'lbfgs', 'sag'], 'C':[ .1, .3, 1, 3]}]

lg = LogisticRegression()
clf = GridSearchCV(lg, parameters, scoring='roc_auc', verbose=1)
clf.fit(X, y)
print("Best parameters set found on training data:")
print(clf.best_params_)
print("Best score found using these parameters:")
print(clf.best_score_)
"""
#%%

print('=== Start cross-validation ====')
lr = LogisticRegression(C=1.0, 
                         penalty='l1', 
                         solver='liblinear',
                         max_iter=500,
                         n_jobs=1)
clf = CalibratedClassifierCV(base_estimator=lr, 
                             method='sigmoid', 
                             cv=5)
scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc', verbose=2)
print('Cross-validation score: {}'.format(scores))
print('Cross-validation score: {}'.format(sum(scores)/5))
print('=== Finished cross-validation ====')


#%%

print('==== Starting fitting ====')
clf.fit(X, y)
print('Starting predicting.')
pred = clf.predict_proba(Xtest)[:,1]
print('==== Finished predicting ====')

#%%

print('==== Creating submission file ====')
submission_id = pd.DataFrame({'id': test_resources["id"]})
submission = pd.concat([submission_id, pd.DataFrame({'project_is_approved': pred})], axis=1)
print(submission.head())
submission.to_csv('submission.csv', index=False)
print('==== All done! Have a nice day! ====')

