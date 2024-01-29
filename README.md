
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
    
## Documentation

For ML model-IsolationForest [Documentation](https://scikit-learn.org/stable/modules/outlier_detection.html#isolation-forest)

For ML model-LocalOutliersFactor [Documentation](https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.)






