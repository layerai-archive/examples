# Credit score model with Layer

[![Open in Layer](https://development.layer.co/assets/badge.svg)](https://development.layer.co/layer/credit-score) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/credit-score/credit-score.ipynb) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/credit-score)

In this project we use Layer to build a credit scoring model. The project uses the [Home Credit Default Risk dataset](https://www.kaggle.com/c/home-credit-default-risk/overview) that is hosted on Kaggle.
## How to use

Make sure you have the latest version of Layer-SDK
```
!pip install layer-sdk -q
```
You can fetch the trained model and start making predictions from it right away. 

```python
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
credit_model = layer.get_model('layer/credit-score/models/credit_score_model').get_train()
data = np.array([[1731690, -1916.0,-1953.0,6953.31,6953.31,1731690,0, 0 ,1731690 ,0.2976,7.47512,0.039812,1731690,0.189752,-161451.0,1731690,1731690,1731690,1731690,1,-16074.0, 1731690, 0.0 ]])
categories = []
transformer = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore', drop="first"), categories)],
        remainder='passthrough')
data = transformer.fit_transform(data)
credit_model.predict(data)
credit_model.predict_proba(data)
# > array([0])
# > array([[0.93264026, 0.06735974]])

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