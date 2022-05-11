# How to integrate Layer and Censius

In this tutorial, we'll look at how to integrate [Layer](layer.ai) and [Censius](http://censius.ai/) in your machine learning projects. 

[![Open in Layer](https://development.layer.co/assets/badge.svg)](https://app.layer.ai/layer/titanic_censius_integ) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/integrations/Censius/Censius_Layer_Integration.ipynb) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/integrations/Censius)

The integration can be done in the following steps. 

## Install layer
```
!pip install -U layer -q
```

## Login into Layer
Next, you need to authenticate your Layer account. 
```
import layer
layer.login()
```
## Import the necessary packages
Next, import the packages we'll need for this project. 
```
from layer.decorators import dataset, model,resources, fabric
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore') 
```
## Initialize your first Layer project
It's time to create your first Layer Project. You can find your created project at https://app.layer.ai

```
layer.init("titanic_censius_integ")

```
## Build passengers dataset
Let's start building our data to train our model. We will be using the Kaggle Titanic Dataset which consists two datasets:

1. train.csv
2. test.csv

Let's clone the Layer Titanic Project repo which has these datasets.

```
!git clone https://github.com/layerai/examples
!mv ./examples/titanic/* ./

```
## Merge and transform data to build the dataset
Next, let do some pre-processing to the dataset.

```
def clean_gender(sex):
    result = 0
    if sex == "female":
        result = 0
    elif sex == "male":
        result = 1
    return result


def clean_age(data):
    age = data[0]
    pclass = data[1]
    if pd.isnull(age):
        if pclass == 1:
            return 37
        elif pclass == 2:
            return 29
        else:
            return 24
    else:
        return age  

@fabric("f-medium")
@dataset("passengers")
@resources(path="./data")
def build_passengers_dataset():
  train_df = pd.read_csv("data/train.csv")
  test_df = pd.read_csv("data/test.csv")
  df = train_df.append(test_df)

  df['Sex'] = df['Sex'].apply(clean_gender)
  df['Age'] = df[['Age', 'Pclass']].apply(clean_age, axis=1)
  df = df.drop(["PassengerId", "Name", "Cabin", "Ticket", "Embarked"], axis=1)

  return df
```
The next step is to build this dataset on Layer.

```
layer.run([build_passengers_dataset])
```
https://app.layer.ai/layer/titanic_censius_integ/datasets/passengers 

## Train the survival model
We will be training a RandomForestClassifier to predict the survivors. As you can see the following function is a simple training function.

```
@model(name='cenius_survival_model')
def train():
    parameters = {
        "test_size": 0.25,
        "random_state": 42,
        "n_estimators": 100
    }
    
    layer.log(parameters)
    df = layer.get_dataset("passengers").to_pandas()
    
    df.dropna(inplace=True)
    X = df.drop(["Survived"], axis=1)
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameters["test_size"], random_state=parameters["random_state"])
    
    random_forest = RandomForestClassifier(n_estimators=parameters["n_estimators"])
    random_forest.fit(X_train, y_train)
    
    y_pred = random_forest.predict(X_test)
    layer.log({"accuracy":accuracy_score(y_test, y_pred)})
    return random_forest

```

Train the model on Layer infrastructure.
```
layer.run([train])
```
https://app.layer.ai/layer/titanic_censius_integ/models/cenius_survival_model
# Load model and data

Next, let load the model and data entities that we just created. 
```
model = layer.get_model('cenius_survival_model').get_train()
df = layer.get_dataset('layer/titanic/datasets/passengers').to_pandas()
passenger = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
data=passenger.sample()
```

# Check the predictions

We can make predictions using the model we just trained. 

```
survival_pred = model.predict(data)
survival_prob = model.predict_proba(data)
survival_prob,survival_pred
# > (array([[0.95, 0.05]]), array([0.]))
```

## Install Censius Client
Let's now install the Censius Client and see how we can use it to register the model and data. 
```
!pip install --trusted-host censius-logs-prod.us-east-1.elasticbeanstalk.com --index-url http://censius-logs-prod.us-east-1.elasticbeanstalk.com:8080/simple/ censius
```


## Create a Censius project


In the Censius console, a **Project** is an organizing mechanism that bunches all datasets and models geared towards solving simliar data related problems. 

It has a unique name, with restricted access to the creator of the project, and whoever they provide accesss to.

#### The Projects Page looks something like this 
![title](https://drive.google.com/uc?export=view&id=1JottyeNkQu6E5hgDawBtibZ09zQk-XLx)
```
import numpy as np
from censius import CensiusClient,DatasetType, ModelType
client = CensiusClient(api_key = "fzifrffbuagjwvlikiflajbygmzyadcd", tenant_id = 'abc')



from censius import CensiusClient,DatasetType, ModelType
response = client.register_project(
    name="CL_titanic",
    icon="random",
    type="Training",
    key="11##$DD12Za"
)
response
# > {'createdAt': '2022-05-10 06:43:05.629564',
 'icon': 'random',
 'key': '11##$DD12Za',
 'leadUserId': 23,
 'name': 'CL_titanic',
 'projectId': 106,
 'type': 'Training'}

```


#### On using the above code to create a new Project **CL_titanic**, the project will appear in the page above 


![title](https://drive.google.com/uc?export=view&id=1McNlfzb10s7p0r8gYVFhsD6ocF7f94KX)

## Register the dataset


Once a project has been registered, datasets can be added to the Project using the ``register_dataset`` API 
with the project ID as one of the inputs. 

The DataFrame passed, is then parsed and readied for model and monitor creation with a handy UI as well. 

```
#Register the dataset
response = client.register_dataset(
        name = 'titanic_dataset',
        file = df.dropna(),
        project_id = 103,
        features = [
            {"name": "Survived", "type": DatasetType.DECIMAL},
            {"name": "Pclass", "type": DatasetType.INT},
            {"name": "Sex", "type": DatasetType.INT},
            {"name": "Age", "type": DatasetType.DECIMAL},
            {"name": "SibSp", "type": DatasetType.INT},
            {"name": "Parch", "type": DatasetType.INT},
            {"name": "Fare", "type": DatasetType.DECIMAL},
        ],
        type=DatasetType.TRAINING_TYPE,
        version="1"
    )
response
# > {'datasetDetails': {'createdAt': '2022-05-10 06:43:10.466501',
  'createdBy': 23,
  'datasetId': 124,
  'features': [{'name': 'Survived', 'type': 'decimal'},
   {'name': 'Pclass', 'type': 'integer'},
   {'name': 'Sex', 'type': 'integer'},
   {'name': 'Age', 'type': 'decimal'},
   {'name': 'SibSp', 'type': 'integer'},
   {'name': 'Parch', 'type': 'integer'},
   {'name': 'Fare', 'type': 'decimal'}],
  'name': 'titanic_dataset',
  'projectId': 103,
  'size': 24037,
  'type': 'Training',
  'version': '1'},
 'fileName': 'titanic_dataset.csv',
 'message': 'Successfully uploaded the file to our server ',
 'processingResponse': 'Successfully sent the dataset for processing',
 'statusCode': 200}
```
#### A) The Datasets tab inside the Project after the API is called. 
![title](https://drive.google.com/uc?export=view&id=1ssFK8Porp98Jw1Jper1xdzLodP5jowEO)



#### B) A sample of the dataset is available on-click. 
![title](https://drive.google.com/uc?export=view&id=112la0KQkWZzdJB5wbohAQf6m66B5qtKS)
# Register the model 
Once the datasets are created, we can use the ``register_model`` API to create a model that is linked to the specific dataset created above. 

These models get listed in the **Models** tab to the left the datasets tab. 
```
response = client.register_model(
    model_id = "titanic_model_int",
    model_name = "titanic_dataset_sdk_actual_v1",
    model_version = "1",
    dataset_id = 123,
    project_id = 103,
    type = ModelType.BINARY_CLASSIFICATION,
    targets = ["Survived"],
    features = ["Pclass", "Sex", "Age","SibSp","Parch","Fare"],
)
print(response)
# > {'ID': 95, 'name': 'titanic_dataset_sdk_actual_v1', 'version': '1', 'createdAt': '2022-05-10 06:43:16.102165', 'type': 'Binary Classification', 'userId': 23, 'target': ['Survived'], 'features': ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'], 'projectId': 103, 'userDefinedModelID': 'titanic_model_int', 'modelUpdateFrequency': '', 'datasetId': 123, 'parentModel': 0, 'window_start_date': 0, 'mappedTableId': 35, 'WindowSize': {'number': 0, 'unit': ''}}

```
#### The Model page has a list of models related to a project/dataset as shown below 


![title](https://drive.google.com/uc?export=view&id=1ORf_42n-KrLfmct19UjXiT7dv8BGhs0B)




#### The Model also comes autoinitialized with performance monitors related to the kind of model that is being registered(classification, regression et al) 


![title](https://drive.google.com/uc?export=view&id=1uISrcXl_1jKfC1d7Qzx5DBIni2F6TKaH)
## Send Production logs to the processed model
Once the model has been registered, and the auto initialized performance monitors are in place, production logs can be flown into the internal databases securely via the ``log`` API, which takes in the log,model ID and a prediction ID (discussed below)

```
import time
import random 

def production_data(df):
    idx = random.randint(0,len(df-1))
    data_point = df.iloc[idx]
    return data_point

def predict_and_log_fn(model,model_id,model_version,features,client,timestamp):
    features_dict = features 
    features = features.to_numpy().reshape(1, -1)
    survival_pred= model.predict(features)
    survival_prob= model.predict_proba(features)
    features_dict = features_dict.to_dict()
    response = client.log(
      model_id=model_id,
      model_version=model_version,
      timestamp = timestamp,
      prediction_id = f"log-{timestamp}",
      features=features_dict,
      prediction = {
      "Survived": {"label": survival_pred.item(), "confidence": np.max(survival_prob)}
    })
    
    return survival_pred.item(),np.max(survival_prob),response 


model_id = "titanic_model_int"
model_version = "1"


x = 5 #no of times you want to simulate your predict function being called
data = df.dropna().drop(columns=['Survived'])
for i in range(x):  
    features = production_data(data)
    timestamp = round(time.time() * 1000)
    pred,prob,response = predict_and_log_fn(model,model_id,model_version,features,client,timestamp)
    time.sleep(5)
```
## Update actuals for the log
Since logs rarely have the actual labels for the production data that is being inserted, we also provide a ``update_actual`` API that uses the unique prediction ID from above to update the actuals column of the log.
```
response = client.update_actual(
    prediction_id = 'loggincheck-1',
    actual = {
        'survived': 1,
    },
    model_id='titanic_model',
    model_version= '1'
)

```
## Console view
After serving the model and sending the prediction logs to Censius, the monitors would automatically appear on the Censius console.

All you need to do is:

1. Log in
2. Click on your specific project
3. Select the model and the model version you are monitoring

## Monitors tab
This tab summarizes all your monitors under one screen. View, analyze, and deep-dive into **unlimited monitors**, their related **violations** along with interactive visuals for each. 

There are 3 types of monitors you can manage through this tab:

*   **Performance monitors**: Tracks performance dips through a wide range of performance metrics 
*   **Drift monitors**: Tracks data drifts and concept drifts
*   **Data quality monitors**: Tracks data quality issues such as missing value, inconsistent data type, data ranges, and more.

You can quickly browse a specific historical time period to track and compare events without going back to code and writing queries every time.

Other details you can navigate here include:

*   Monitor ID
*   Trigger condition
*   Monitor type
*   Date of last violation
*   Violation severity

![title](https://drive.google.com/uc?export=view&id=1dehzJ4XbjfgD0EI8LScVpL7zUhc4E83x)
## Performance tab

This tab has two tools:

1. Traffic comparison tool to see the gaps between predictions and actuals
2. Performance comparison line tool to pin-point the relationships between two or more metrics
### Traffic Comparison

![title](https://drive.google.com/uc?export=view&id=1cO2p9Asn5Qzx7xoAoLn2EzoH6dDR3gPd)

### Performance Comparison

![title](https://drive.google.com/uc?export=view&id=1wHeSVuxTMIfGt5eXHll3CbaMquHxUCxu)
## Dashboards

Dashboards allows you to investigate the violations further and analyze the root cause through various visualization and analysis tools including tables, line graphs, histograms, numeric functions such as count, and much more

![title](https://drive.google.com/uc?export=view&id=1N7mnyzGTfBi3KnzQngSvLMuiYWp0SHWg)
 Start **collaborating** with [Layer](https://layer.ai) and **monitoring** with [Censius](https://censius.ai/get-started) within 5 minutes!
## Where to go from here?

Now that you have created first Layer Project, you can:

- Join our [Slack Community ](https://bit.ly/layercommunityslack)
- Visit [Layer Examples Repo](https://github.com/layerai/examples) for more examples
- Browse [Trending Layer Projects](https://layer.ai) on our mainpage
- Check out [Layer Documentation](https://docs.layer.ai) to learn more
