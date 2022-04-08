# Credit score model with Layer

[![Open in Layer](https://development.layer.co/assets/badge.svg)](https://development.layer.co/layer/credit-score) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/credit-score/credit-score.ipynb) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/credit-score)
## How to use

Make sure you have the latest version of Layer-SDK
```
!pip install layer-sdk -q
```
You can fetch the trained model and start making predictions from it right away. 

```python
from sklearn.model_selection import train_test_split
import layer
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
application_features =  layer.get_dataset('layer/credit-score/datasets/application_features').to_pandas()
previous_application_features = layer.get_dataset('layer/credit-score/datasets/previous_application').to_pandas()
installments_payments = layer.get_dataset('layer/credit-score/datasets/installments_payments').to_pandas()
dff = installments_payments.merge(previous_application_features, on=['SK_ID_PREV', 'SK_ID_CURR']).merge(application_features,on=['SK_ID_CURR'])
X = dff.drop(["TARGET", "SK_ID_CURR",'index'], axis=1)
y = dff["TARGET"]
credit_model = layer.get_model('layer/credit-score/models/credit_score_model').get_train()
categories = X.select_dtypes(include=['object']).columns.tolist() 
transformer = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore', drop="first"), categories)],
        remainder='passthrough')

X = transformer.fit_transform(X,)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=0)
credit_model.predict(X_test[0:1])
credit_model.predict_proba(X_test[0:1])
# > array([0])
# > array([[0.86409123, 0.13590877]])
```
## Dataset
In this project we build a credit scoring model using the 
[Home Credit Default Risk dataset](https://www.kaggle.com/c/home-credit-default-risk/overview) 
that is hosted on Kaggle.

https://development.layer.co/layer/credit-score/datasets/application_features

https://development.layer.co/layer/credit-score/datasets/installments_payments
## Model 
We use the [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
that is ideal for big dataset. According to its documentation: 

This estimator has native support for missing values (NaNs). During training, the tree grower learns at each split point whether samples with missing values should go to the left or right child, based on the potential gain. When predicting, samples with missing values are assigned to the left or right child consequently. If no missing values were encountered for a given feature during training, then samples with missing values are mapped to whichever child has the most samples.

This implementation is inspired by [LightGBM](https://github.com/Microsoft/LightGBM).

Check out the model on Layer on the link below:

https://development.layer.co/layer/credit-score/models/credit_score_model