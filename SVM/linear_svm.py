#import lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

#import dataset
dataset= pd.read_csv('Social_Network_ads.csv')
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

#splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.25, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sx=StandardScaler()
x_train=sx.fit_transform(x_train)
x_test=sx.transform(x_test)

#classifier
from sklearn.svm import SVC
classifier=SVC(kernel='linear', random_state=0)
classifier.fit(x_train,y_train)

#prediction
y_pred= classifier.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)