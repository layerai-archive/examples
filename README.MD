# Layer Example Projects

This repository contains example projects that you can use to get started with [Layer](https://layer.ai).

Layer is a collaborative MLOps platform where you can build, train, version and share your machine learning (ML) models.

## Install Layer

The first step is to install [Layer](https://docs.layer.ai/docs/installation):

```yaml
pip install layer
```

## Clone the examples repository

The first step is to clone this repository:

```yaml
git clone https://github.com/layerai/examples
```

## Select an example project

Next, select one example project and change into that folder. Let's use the Titanic example for illustration:

```yaml
cd examples/titanic
```

Open the associated notebook or Python script and run it. Layer runs your project and places the generated entities in the appropriate Discover tabs.

## Use the generated entities in a Jupyter Notebook

Entities generated with Layer can also be accessed in a Jupyter Notebook. Layer allows you to access the datasets, feature sets, and models.

First, let's look at how to access the created datasets:

```python
import layer
dataset = layer.get_dataset('layer/titanic/datasets/passengers')
```

The model can be accessed using Layer `get_model` function:

```python
import layer
model = layer.get_model('layer/titanic/models/survival_model')
```

The model can be used to make predictions right away:

```python
df = layer.get_dataset("passengers").to_pandas()
passenger = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
survival_probability = model.get_train().predict_proba(passenger.sample())[0][1]
print(f"Survival Probability: {survival_probability:.2%}")

# > Survival Probability: 68.37%
```

## Next steps

To learn more about using layer, you can:

- Join our [Slack Community ](https://bit.ly/layercommunityslack)
- Visit [Layer Examples Repo](https://github.com/layerai/examples) for more examples
- Browse [Trending Layer Projects](https://layer.ai) on our mainpage
- Check out [Layer Documentation](https://docs.layer.ai) to learn more
