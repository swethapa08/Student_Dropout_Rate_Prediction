import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings
import pickle

# Data pre-processing
from sklearn.preprocessing import StandardScaler

# Data splitting
from sklearn.model_selection import train_test_split

# Machine learning Models
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import sklearn.linear_model._logistic

# Evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc

data=pd.read_csv("D:\swetha\education-website-main\education-website-main\dataset.csv")

data=np.array(data)

x=data[1:,1:-1]
y=data[1:,-1]
#y=y.astype('int')
#x=x.astype('int')
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
log_reg = LogisticRegression()


log_reg.fit(X_train, y_train)

inputt=[int(x) for x in "6 1 1 1 1 0 1 0 0 0 7.5 8.0".split(' ')]
final=[np.array(inputt)]

b = log_reg.predict_proba(final)


pickle.dump(log_reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
