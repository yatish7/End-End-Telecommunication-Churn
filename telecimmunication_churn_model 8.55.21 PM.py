
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import sklearn
print(sklearn.__version__)

import sys
print(sys.version)

df=pd.read_csv("tel_churn.csv")
df.head()

df=df.drop('Unnamed: 0',axis=1)

x=df.drop('Churn',axis=1)
x.head()
x.info()

y=df['Churn']
y.head()
y.info()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier

model_rf=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)

model_rf.fit(x_train,y_train)

y_pred=model_rf.predict(x_test)

model_rf.score(x_test,y_test)

print(classification_report(y_test, y_pred, labels=[0,1]))

from imblearn.combine import SMOTEENN
sm = SMOTEENN()
X_resampled1, y_resampled1 = sm.fit_resample(x, y)

xr_train1,xr_test1,yr_train1,yr_test1=train_test_split(X_resampled1, y_resampled1,test_size=0.2)

model_rf_smote=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)

model_rf_smote.fit(xr_train1,yr_train1)

yr_predict1 = model_rf_smote.predict(xr_test1)

model_score_r1 = model_rf_smote.score(xr_test1, yr_test1)

print(model_score_r1)
print(metrics.classification_report(yr_test1, yr_predict1))

print(metrics.confusion_matrix(yr_test1, yr_predict1))

"""### **PCA**"""

from sklearn.decomposition import PCA
pca = PCA(0.9)
xr_train_pca = pca.fit_transform(xr_train1)
xr_test_pca = pca.transform(xr_test1)
explained_variance = pca.explained_variance_ratio_

model=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)

model.fit(xr_train_pca,yr_train1)

yr_predict_pca = model.predict(xr_test_pca)

model_score_r_pca = model.score(xr_test_pca, yr_test1)

print(model_score_r_pca)
print(metrics.classification_report(yr_test1, yr_predict_pca))

"""NO better results with PCA so consider the model we got previously"""

import pickle
filename = 'model.sav'
pickle.dump(model_rf_smote, open(filename, 'wb'))
load_model = pickle.load(open(filename, 'rb'))
model_score_r1 = load_model.score(xr_test1, yr_test1)
print(model_score_r1)