# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 22:08:12 2020

@author: DELL
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import pickle

data = pd.read_csv(r'data2.csv')
x = data.iloc[:, 0:9]
y = data.variables
y = y.map({'recurrence-events':1,'no-recurrence-events':0})

data.age.replace(['10-19', '20-29', '30-39', '40-49','50-59', '60-69', '70-79', '80-89', '90-99'], [1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
data.variables.replace(['recurrence-events','no-recurrence-events'], [1, 0], inplace=True)
data.menopause.replace(['premeno','lt40', 'ge40'], [1, 2, 3], inplace=True)
data.tumorsize.replace(['0-4','5-9','10-14','15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59'], [1, 2,3,4,5,6,7,8,9,10,11,12], inplace=True)
data.nodecaps.replace(['yes','no'], [1, 2], inplace=True)
data.breast.replace(['right','left'], [1, 2], inplace=True)
data.breastquad.replace(['left_up','left_low','right_up','right_low','central'], [1, 2,3,4,5], inplace=True)
data.irradiat.replace(['yes','no'], [1, 0], inplace=True)
data.invnodes.replace(['0-2','3-5','6-8','9-11','12-14','15-17','18-20','21-23','24-26','27-29','30-32','33-35'], [1, 2,3,4,5,6,7,8,9,10,11,12], inplace=True)

y = np.array(data.variables.tolist())
data = data.drop('variables', 1)
X = np.array(data.to_numpy())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
params = {'criterion':['entropy'],
          'n_estimators':[10],
          'min_samples_leaf':[1],
          'min_samples_split':[3], 
          'random_state':[123],
          'n_jobs':[-1]}
model1 = GridSearchCV(classifier, param_grid=params, n_jobs=-1)
model1.fit(X_train,y_train)
pickle.dump(model1,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
