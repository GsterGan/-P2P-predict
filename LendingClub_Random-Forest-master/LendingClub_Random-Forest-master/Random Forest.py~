#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 19:38:56 2018

@author: shilinli
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
loans = pd.read_csv('loan_data.csv')

# Fico Dist w/ Credit Policy
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')

# Fico Dist w/ not.fully.paid
plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Not.Fully.Paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Not.Fully.Paid=0')
plt.legend()
plt.xlabel('FICO')

# Count plot on Purpose seperated by not.fully.paid
plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')

# Trend between FICO and Interest Rate
sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')

# Trend differed between not.fully.paid and credit.policy
plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
           col='not.fully.paid',palette='Set1')


# Transform Categorical to Dummies
final_dt = pd.get_dummies(loans,columns=['purpose'],drop_first = True) 


# Train Test Split
from sklearn.cross_validation import train_test_split
X = final_dt.drop('not.fully.paid',axis = 1)
Y = final_dt['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

# Train a Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# Predictions and Evaluation of Decision Tree
pred = dtree.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(y_test,pred)
classification_report(y_test,pred)


# Training the Random Forest model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)

# Predictions and Evaluation of Random Forest
rfc_pred = rfc.predict(X_test)
confusion_matrix(y_test,rfc_pred)
classification_report(y_test,rfc_pred)


