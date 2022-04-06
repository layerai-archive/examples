# Getting started with Layer

[![Open in Layer](https://development.layer.co/assets/badge.svg)](https://development.layer.co/layer/titanic) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/titanic/GettingStartedWithLayer.ipynb) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/titanic)

In this quick walkthrough, we will train a machine learning model to predict the survivors of the Titanic disaster and deploy it for real-time inference using Layer.

## How to use

Make sure you have the latest version of Layer-SDK
```
!pip install layer-sdk -q
```

```python
import layer

model = layer.get_model("survival_model").get_train()
df = layer.get_dataset("passengers").to_pandas()
passenger = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
survival_propability = model.predict_proba(passenger.sample())[0][1]

print(f"Survival Probaility: {survival_propability:.2%}")

# > 0.68
```

## Dataset

We will use the famous [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic/data) to train our model. 


## Model

We will be train a RandomForestClassifier for sklearn.