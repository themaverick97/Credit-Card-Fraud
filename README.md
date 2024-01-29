
# FindDefault (Prediction of Credit Card fraud)

A credit card is one of the most used financial products to make online purchases and
payments. Though the Credit cards can be a convenient way to manage your finances, they can
also be risky. Credit card fraud is the unauthorized use of someone else&#39;s credit card or credit
card information to make purchases or withdraw cash.
It is important that credit card companies are able to recognize fraudulent credit card
transactions so that customers are not charged for items that they did not purchase.


## Appendix

- Importing the required libraries
- Exploratory Data Analysis(EDA)
- Data sampling
- Scaling
- Selecting the Independent and Dependent Variable
- Spliting Dataset into Train and Test
- Dealing with Imbalanced Data
- Training the ML Model
- Model validation
## Importing

Importing libraries 

```bash
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_auc_score,roc_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
```
    
## EDA

![image](https://github.com/themaverick97/Credit-Card-Fraud/assets/121721006/c9febff6-e6c1-4c20-8d5f-ab7259a648c6)

![image](https://github.com/themaverick97/Credit-Card-Fraud/assets/121721006/bd170851-562a-48ed-998d-7361cf54cc2d)

![image](https://github.com/themaverick97/Credit-Card-Fraud/assets/121721006/1d7fc27c-cbe4-4854-9ead-9a49d080f247)




## Sampling

Samples are used to make inferences about populations. Samples are easier to collect data from because they are practical, cost-effective, convenient, and manageable.

```bash
df1= df.sample(frac = 0.1,random_state=1)
df1.shape
```

## Scaling
In this project i have used Robust.Scale features using statistics that are robust to outliers.This Scaler removes the median and scales the data according to the quantile range (defaults to IQR: Interquartile Range). The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).

```bash
rs=RobustScaler()
df1['Amount']=rs.fit_transform(df1['Amount'].values.reshape(-1,1))
df1['Time']=rs.fit_transform(df1['Time'].values.reshape(-1,1))
```
## Selecting the Independent and Dependent Variable
In this project 'Time','Amount' are the independent and 'Class'are the Dependent Variable
```bash
#Create independent and Dependent Features
X=df1.iloc[:,:-1]
y=df1['Class']
```
## Spliting Dataset into Train and Test
By using the train test split,we split the data into X_train,X_test,y_train,_y_test
```bash
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)
```
## Dealing with Imbalanced Data
As the data given was higly Imbalanced.In this I am using SMOTE(overfiiting)method
```bash
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train,y_train)
```
## Model Training
In Model Traing I have used Randomforest,LogisticRegression,LocalOutlierFactor,IsolationForest for the best Model
## Randomforest
```bash
rf_clf = RandomForestClassifier()
param_grid = {
    'n_estimators': [50,100,200],
    'criterion':['gini'],
    'min_samples_split': [2, 5],
    'max_features': [ 'sqrt', 'log2']
}
grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_smote, y_train_smote)
```
## Using  LogisticRegression
```bash
model=LogisticRegression()
model.fit(X_train,y_train)
```
## Using IsolationForest
```bash
"Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X),
                                       contamination=outlier_fraction,random_state=42, verbose=0)
```
## Using LocalOutlierFactor
```bash
"Local Outlier Factor":LocalOutlierFactor(n_neighbors=20, algorithm='auto',
                                              leaf_size=30, metric='minkowski',
                                              p=2, metric_params=None, contamination=outlier_fraction)
```
## Model validation
For the model validation I used the Accuracy and the Classfication Report
```bash
accuracy = accuracy_score(y_test, y_pred_lr)
print("Accuracy on the test set:", accuracy)
cr=classification_report(y_test,y_pred_lr)
print(cr)
n_errors=(y_pred_lr !=y_test).sum()
print("{}: {}".format(model,n_errors))
```
## Make a prediction on Fraud or Valid
As give values to the Function and it predicts the transcation if it Fraud or valid
```bash
def pred(data):
    data = np.asarray(data).reshape(1,-1)
    predd = model.predict(data)
    if predd == 0:
        print("it's a Valid Transcation")
    else:
        print("it's a fraudulent Transcation")
```
## Documentation

For ML model-IsolationForest [Documentation](https://scikit-learn.org/stable/modules/outlier_detection.html#isolation-forest)

For ML model-LocalOutliersFactor [Documentation](https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.)









