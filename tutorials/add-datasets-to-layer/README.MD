# How to add dataset to Layer.
[![Open in Layer](https://development.layer.co/assets/badge.svg)](https://app.layer.ai/layer/iris/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/tutorials/add-datasets-to-layer/how_to_add_dataset_to_layer.ipynb) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/tutorials/add-datasets-to-layer)

This tutorial shows how to create a dataset in Layer. We use the famous Iris dataset to illustrate this. 

In Layer datasets are created using the `dataset` decorator. Checkout the attached notebook to learn more. 

```
@dataset('iris_data')
def save_iris():
  from sklearn import datasets
  import pandas as pd
  iris = datasets.load_iris()
  df = pd.DataFrame(data=iris.data, columns = iris.feature_names)
  return df
  layer.run([save_iris])
```
https://app.layer.ai/layer/iris/
