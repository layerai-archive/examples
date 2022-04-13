# How to add models to Layer
[![Open in Layer](https://development.layer.co/assets/badge.svg)](https://app.layer.ai/layer/iris/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/add-models-to-layer/how_to_add_models_to_layer.ipynb) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/add-datasets-to-layer)

In this project, we'll look at how to save a model to Layer. Layer supports popular machine learning libraries and frameworks. 
## How to use
Make sure you have the latest version of Layer-SDK

```
!pip install layer-sdk -q 
```
You can fetch the trained model and start making predictions from it right away.

```python
import layer
dataset = layer.get_dataset('layer/iris/datasets/iris_data').to_pandas()
model = layer.get_model('layer/iris/models/iris_model').get_train()
model.predict(dataset[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].head(1))
# > array(['Iris-setosa'], dtype=object)
```
## Dataset 
In this project, we use the Iris dataset to demonstrate how to 
build and save models to Layer.
https://app.layer.ai/layer/iris/datasets/iris_data
## Model 
We train a simple [Support Vector Machine](https://scikit-learn.org/stable/modules/svm.html) model and save it to Layer. 
https://app.layer.ai/layer/iris/models/iris_model