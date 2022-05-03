# California house price prediction 
[![Open in Layer](https://development.layer.co/assets/badge.svg)](https://app.layer.ai/layer/california_housing) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/housing/housing.ipynb) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/housing)

We used the California housing dataset to build a regression model for predicting the price of a house. 

## Dataset
The California housing is a dataset obtained from the StatLib repository. Here is the included description:

S&P Letters Data
We collected information on the variables using all the block groups in California from the 1990 Cens us. In this sample a block group on average includes 1425.5 individuals living in a geographically co mpact area. Naturally, the geographical area included varies inversely with the population density. W e computed distances among the centroids of each block group as measured in latitude and longitude. W e excluded all the block groups reporting zero entries for the independent and dependent variables. T he final data contained 20,640 observations on 9 variables. The dependent variable is ln(median house value).

The file contains all the variables. Specifically, it contains median house value, med ian income, housing median age, total rooms, total bedrooms, population, households, latitude, and lo ngitude in that order.
 

[source](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)

## How to use 

Ensure you have the latest versio of Layer: 
```
pip install layer --upgrade
```

Fetch the data as follows: 
```
import layer

dataset = layer.get_dataset('layer/california_housing/datasets/train').to_pandas()
dataset.head()

```

https://app.layer.ai/layer/california_housing/datasets/train https://app.layer.ai/layer/california_housing/datasets/test
## Model
We have trained a linear regression model that can be fetched and used to make predictions. 

```
import numpy as np
test = layer.get_dataset('layer/california_housing/datasets/train').to_pandas()
model = layer.get_model('layer/california_housing/models/housing').get_train()
x = test.drop('median_house_value', axis=1)
model.predict(x.head(1))
# > array([352812.31112454])
```

https://app.layer.ai/layer/california_housing/models/housing

## Reference
```
Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions, Statistics and Probability Letters, 33 (1997) 291-297.
```