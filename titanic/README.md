# Getting started with Layer

[![Open in Layer](https://app.layer.ai/assets/badge.svg)](https://app.layer.ai/layer/titanic) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/titanic/Getting_Started_With_Layer.ipynb) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/titanic)

In this quick walkthrough, we will train a machine learning model to predict the survivors of the Titanic disaster and deploy it for real-time inference using Layer.

## How to use

Make sure you have the latest version of Layer:
```
!pip install layer -q
```

```python
import layer

model = layer.get_model('layer/titanic/models/survival_model').get_train()
df = layer.get_dataset('layer/titanic/datasets/passengers').to_pandas()
passenger = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
survival_probability = model.predict_proba(passenger.sample())[0][1]
print(f"Survival Probability: {survival_probability:.2%}")

# > Survival Probability: 68.37%
```

## Dataset

We will use the famous [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic/data) to train our model. This dataset originally contains two separate files `train.csv` and `test.csv`. We are going to create a new dataset by merging and transforming them. Here is the final dataset:

https://app.layer.ai/layer/titanic/datasets/passengers

## Model

We will be training a RandomForestClassifier from sklearn. We will fit the dataset we have created. You can find all the model experiments here:

https://app.layer.ai/layer/titanic/models/survival_model
