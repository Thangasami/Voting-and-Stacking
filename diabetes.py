# -*- coding: utf-8 -*-
"""
Created on Sat May  8 08:32:14 2021

@author: sthan
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from mlxtend.classifier import StackingCVClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings(('ignore'))



df = pd.read_csv('diabetes.csv')

X = df.drop('diabetes', 1)
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y )


df['diabetes'].value_counts()

#create a KNN Model

knn = KNeighborsClassifier()
params_knn = {'n_neighbors': np.arange(1, 25)}
knn_gs = GridSearchCV(knn, params_knn, cv = 5)
knn_gs.fit(X_train, y_train)

knn_best = knn_gs.best_estimator_
print(knn_best)


#create a RFC model

rf = RandomForestClassifier()
param_rf = {'n_estimators': [50,100, 200 ]}
rf_gs = GridSearchCV(rf, param_rf, cv = 5)

rf_gs.fit(X_train, y_train)

rf_best = rf_gs.best_estimator_
print(rf_gs.best_params_)
print(rf_gs.best_estimator_)


#create a Logistic regression Model

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

#Test Accuracy
print('knn {}'.format(knn_best.score(X_test, y_test)))
print('rf {}'.format(rf_best.score(X_test, y_test)))
print('logisticRegression {}'.format(log_reg.score(X_test, y_test)))

#create a dict for our models.

estimators = [('knn', knn_best), ('rf', rf_best), ('log_reg', log_reg)]

#Voting Classifier
vc = VotingClassifier(estimators, voting='hard')
vc.fit(X_train, y_train)
vc.score(X_test, y_test)


#stacking
clf1 = KNeighborsClassifier(n_neighbors = 10)
clf2 = GaussianNB()
clf3 = RandomForestClassifier(random_state=46)
lr = LogisticRegression()

sclf =  StackingCVClassifier(classifiers = [clf1, clf2, clf3], meta_classifier = lr, random_state=46)

sclf.fit(X_train, y_train)
a = sclf.score(X_test, y_test)
print(a)
sclf =  StackingCVClassifier(classifiers = [clf1, clf2, clf3], meta_classifier = lr, random_state=46, use_probas = True)


sclf.fit(X_train, y_train)
b = sclf.score(X_test, y_test)


print(b)


variance_inflation_factor(X_train.values, 0)

for i in range(len(X_train.columns)):
    print(variance_inflation_factor(X_train.values, i))



































