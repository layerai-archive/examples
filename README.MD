# Layer Example Projects

This repository contains example projects that you can use to get started with [Layer](https://layer.ai).

Layer is a declarative MLOps platform to help data teams across all companies produce machine learning (ML) applications based on code. It orchestrates your data pipeline for you. Layer enables you to focus on designing, developing, and deploying models without worrying about the infrastructure

## Install Layer
The first step is to install the [Layer SDK](https://docs.app.layer.ai/docs/installation):
```yaml
pip install layer-sdk
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

## Deploying the model 
Once the model runs successfully, Layer will output links to the created entities. 
You can deploy your model by clicking the Deploy button next to the model in the Models tab. 

Once the deployment process is complete, you will get an
API end point that you can use make predictions on the model. 

## Run predictions on the deployed model 
Click the 🔗 icon to copy the deployment URL. You can then use this URL to run predictions on the model: 

```yaml
curl --header "Content-Type: application/json; format=pandas-records" \
  --request POST \
  --data '[{"Sex":0, "EmbarkStatus":0, "Title":0, "IsAlone":0, "AgeBand":1, "FareBand":2}]' \
  $MODEL_DEPLOYMENT_URL
# > [1.0]
```

The API returns either a [0] or a [1], predicting whether that passenger survived.

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
To learn more about using layer, checkout the [official docs](https://docs.app.layer.ai/docs/). 

You can also join our [community on Slack](https://join.slack.com/t/layer-community/shared_invite/zt-o3cneq4y-5lGSZH~9NhOmBWQB4WAsJw) to learn and engage with other MLOps enthusiasts. 