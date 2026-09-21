# -*- coding: utf-8 -*-
"""Breast cancer classification with logistic regression.

Script export of breast_cancer_logistic_regression.ipynb (originally built in
Google Colab). Install dependencies first:

    pip install pandas numpy scikit-learn ucimlrepo

# Logistic Regression

## Importing the libraries
"""

import pandas as pd
from ucimlrepo import fetch_ucirepo

"""## Importing the dataset"""

# fetch dataset from UC Irvine ML Repository https://archive.ics.uci.edu/
breast_cancer_wisconsin_original = fetch_ucirepo(id=15)

# data (as pandas dataframes)
X = breast_cancer_wisconsin_original.data.features
y = breast_cancer_wisconsin_original.data.targets

#Preprocessing

# NaN in X
print(X.isnull().sum())

# Applying Imputation for the X on
from sklearn.impute import SimpleImputer
import numpy as np
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X)
X = imputer.transform(X)

"""## Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""## Feature Scaling (Optional)

"""

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train) # Fit and transform the training data
# X_test = sc.transform(X_test)       # Only transform the test data

"""## Training the Logistic Regression model on the Training set"""

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

"""## Predicting the Test set results"""

y_pred = classifier.predict(X_test)
y_pred

"""## Making the Confusion Matrix"""

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

"""## Computing the accuracy with k-Fold Cross Validation"""

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train.values.ravel(), cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

"""After Feature Scaling :  Accuracy: 96.43 %
Standard Deviation: 2.52 %

Before  Feature Scaling :Accuracy: 96.60 %
Standard Deviation: 2.58 %

So we will avoid feature scaling
"""