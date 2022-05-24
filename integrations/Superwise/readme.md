# 🚀 Getting started with Superwise.ai and Layer.ai on AWS Sagemaker
[![Open in Layer](https://development.layer.co/assets/badge.svg)](https://app.layer.ai/layer/superwise) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/integrations/Superwise/superwise.ipynb) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/integrations/Superwise)

In this notebook, we will demonstrate how to integrate a Sagemaker based development workflow with Superwise.ai and Layer.ai. 

Part I of this notebook walks you through fetching a pre-trained model from Layer. 

Part II of this notebook will walk you through how to setup Superwise.ai to start tracking your Layer model, by registering and providing a baseline for the model's behavior.

Part III will demonstrate how to send new predictions from your model to Superwise.ai, simulating a post-deployment scenario.

At this point, you should be able to start seeing insights from Superwise.ai in the web portal.

### 📌 Prerequisites

1. Be familiar with AWS's SageMaker
2. A Superwise.ai account that enables you to login and view insights
3. A set of API keys for sending data to Superwise.ai 
4. Permissions to create models, training jobs and inference endpoints inside Sagemaker
5. Grant Superwise.ai permissions to your SageMaker bucket #soon to be removed

Note: this notebook works best when run from within a [Sagemaker's notebook](https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html) instance

```
pip install sagemaker layer superwise
```

```
import os
import datetime
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import boto3
from sagemaker import get_execution_role
import sagemaker


sm_boto3 = boto3.client("sagemaker")

sess = sagemaker.Session()
region = sess.boto_session.region_name
bucket = sess.default_bucket()  # this could also be a hard-coded bucket name

print("Using bucket " + bucket)
# > Using bucket sagemaker-us-east-1-341398874395

```
## 🏗️ Part I - Fetching the housing model from Layer

This is a classical LinearRegression model, that uses a publicly available dataset.

This guide is based on the best practices from [AWS Sagemaker's example for building a Scikit-Learn model](https://github.com/aws/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/scikit_learn_randomforest/Sklearn_on_SageMaker_end2end.ipynb)
### 📌 Prerequisites

1. Be familiar with AWS's SageMaker
2. A Superwise.ai account that enables you to login and view insights
3. A set of API keys for sending data to Superwise.ai 
4. Permissions to create models, training jobs and inference endpoints inside Sagemaker
5. Grant Superwise.ai permissions to your SageMaker bucket #soon to be removed

Note: this notebook works best when run from within a [Sagemaker's notebook](https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html) instance

```
%env AWS_DEFAULT_REGION=YOUR_AWS_DEFAULT_REGION
%env AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY_ID
%env AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_ACCESS_KEY

```
### Import pre-trained Layer model 
We'll use a model that's already trained on Layer. To train your own model from scratch check out our [Quickstart notebook](https://github.com/layerai/examples/tree/main/titanic). 

```
from layer.decorators import dataset, model,resources
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import layer
my_model = layer.get_model("layer/california_housing/models/housing").get_train()
```
### Persist the model
```
# persist model
import os
import joblib
path = os.path.join(".", "model.joblib")
joblib.dump(my_model, path)
print("model persisted at " + path)
# > model persisted at ./model.joblib

```
### Local inference
Let's fetch the training data and test the model. 
```
train = layer.get_dataset("layer/california_housing/datasets/train").to_pandas()
X = train.drop(columns="median_house_value")
y = train["median_house_value"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

Test the inference flow:
1. load model
2. process raw input (serialized numpy array)
3. predict and return results 

**Note** 

Deploying a model with SageMaker requires a script containing the following functions: 

- model function named `model_fn`
- input function named `input_fn`
- predictio function named `predict_fn`

The function is attached on this [repo](https://github.com/layerai/examples/tree/main/integrations/Supwerwise/my_script.py). 

We attach a unique ID per instance for the prediction.
For the purpose of demonstration, we use the Dataframe index as the ID of the instances.

In a production setting, you should use an ID that has semantic meaning in the context of your application, such as transaction_id etc.
```
import my_script
local_model = my_script.model_fn(".")
from io import BytesIO
rows = X_train.head(10)
#convert the Dataframe index to a string ID
indexes = rows.index.map(str)
inference_input = rows.to_numpy()

#attach the ID to the payload
inference_input = np.insert(inference_input, 0, indexes, axis=1)
np_bytes = BytesIO()
# Serialize the payload
np.save(np_bytes, inference_input, allow_pickle=True)
input_data = my_script.input_fn(np_bytes.getvalue(), "application/x-npy")
predictions = my_script.predict_fn(input_data, local_model)
# > array([[  9173.        , 131330.90954797],
       [ 16528.        ,  89092.75958676],
       [  8791.        , 291806.28006424]])
```

### 🚀 Deploy to a real-time endpoint
Create a Sagemaker *Model* from s3 artifacts, and deploy it to an Endpoint
```
#Build tar file with model data + inference code
import subprocess
bashCommand = "tar -cvpzf model.tar.gz model.joblib"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
#Bucket for model artifacts
default_bucket = sess.default_bucket()
print(default_bucket)
boto_session = boto3.session.Session()
s3 = boto_session.resource('s3')
#Upload tar.gz to bucket
model_artifacts = f"s3://{default_bucket}/model.tar.gz"
response = s3.meta.client.upload_file('model.tar.gz', default_bucket, 'model.tar.gz')
from sagemaker.sklearn.model import SKLearnModel
model = SKLearnModel(
    name='housing-superwise',
    model_data=model_artifacts,
    role=get_execution_role(),
    entry_point='my_script.py',
    framework_version='0.23-1')
predictor = model.deploy(instance_type="ml.t2.medium", initial_instance_count=1)

# > Using already existing model: housing-superwise
# >---------!

```

✅ Test the Endpoint by running an Inference request
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
train = pd.read_csv("train.csv")
X = train.drop(columns="median_house_value")
y = train["median_house_value"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Prepare the input as a numpy array with record_id, feature 1... feature n

# Treat the Dataframe index as the record_id
indexes = X_train.index.map(str)
X_train_inference = X_train.to_numpy()
X_train_inference = np.insert(X_train_inference, 0, indexes, axis=1)
predictions = predictor.predict(X_train_inference)

# Returns a column with ID and the prediction value
predictions
# >array([[  9173.        , 131330.90954797],
       ...,
       [ 15795.        , 199673.75455909]])
```

## 📈 Part II - Tracking the Layer model with Superwise.ai


```
import os
os.environ["SUPERWISE_CLIENT_ID"] = 'YOUR_SUPERWISE_CLIENT_ID'
os.environ["SUPERWISE_SECRET"]='YOURSUPERWISE_SECRET'

from superwise import Superwise
from superwise.models.model import Model
from superwise.models.version import Version
from superwise.models.data_entity import DataEntity
from superwise.resources.superwise_enums import FeatureType, DataEntityRole
sw = Superwise()
```

### Create the Model entity
```
housing_model = Model(
    name="housing-layer-and-superwise",
    description="4.0"
)
my_model = sw.model.create(housing_model)
print(my_model.get_properties())
# > {'active_version_id': None, 'description': '4.0', 'external_id': '60', 'id': 60,
#  'is_archived': False, 'name': 'housing-layer-and-superwise', 'time_units': ['D', 'W']}
```

### Add the prediction value, a timestamp and the label to the training features

```
baseline_data = X_train.assign(
    prediction=predictions[:,1].astype(float),
    ts=pd.Timestamp.now(),
    median_house_value=y_train
)

# treat the Dataframe index as a record ID - for demonstration purpose only. 
baseline_df = baseline_data.reset_index().rename(columns={"index": "record_id"})

```
### Create a *Schema* object that describes the format and sematics of our Baseline data

The Schema object helps Superwise.ai interpret our data, for example - undertand which column prepresents predictions and which represents the labels.


```
entities_collection = sw.data_entity.summarise(
    data=baseline_df,
    specific_roles = {
      'record_id': DataEntityRole.ID,
      'ts': DataEntityRole.TIMESTAMP,
      'prediction': DataEntityRole.PREDICTION_VALUE,
      'median_house_value': DataEntityRole.LABEL
    }
)

```
Here is the schema main properties (roles, types, feature importance and descriptive statistics):
```
ls = list()
for entity in entities_collection:
    ls.append(entity.get_properties())
    
pd.DataFrame(ls).head()
```

### Create a *Version* object

As explained above, a *Version* represents a concrete ML model we are tracking.

A *Version* solves a *Model*

A *Version* has a *Baseline*


```
housing_version = Version(
    model_id=my_model.id,
    name="5.0",
    data_entities=entities_collection,
)

my_version = sw.version.create(housing_version)
sw.version.activate(my_version.id)
# <Response [200]>

```
### ✅  Verifying the setup
![Verifying the setup](https://github.com/layerai/examples/raw/main/integrations/Superwise/images/image1.png)
## 🩺 Part III - monitoring ongoing predictions

Now that we have a *Version* of the model setup with a *Baseline*, we can start sending ongoing model predictions to Superwise to monitor the model's performance in a production settings.

For this demo, we will treat the Test split of the data as our "ongoing predictions".

```
# insert the record ID as part of the input payload to the model
indexes = X_test.index.map(str)
X_test_inference = X_test.to_numpy()
X_test_inference = np.insert(X_test_inference, 0, indexes, axis=1)
predictions = predictor.predict(X_test_inference)
predictions
# array([[ 10941.        , 143357.7423556 ],
       ...,
       [  1275.        , 154390.03384507]])

# Note: we provide the column names we declared in the Schema object, 
# so that Superwise.ai will be able to interpret the data

ongoing_predictions = pd.DataFrame(data=predictions, columns=["record_id", "prediction"])
ongoing_predictions["record_id"] = ongoing_predictions["record_id"].astype(int)

ongoing_features = X_test.copy()
ongoing_features['record_id'] = X_test.index
ongoing_data = ongoing_features.merge(ongoing_predictions, how='left', on='record_id')
ongoing_data['ts'] = pd.Timestamp.now()

```
### Log the production data in Superwise
```
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
# Log the production data in Superwise (Max chunck size = 1000 predictions)

ongoing_data_chuncks = chunks(ongoing_data.to_dict(orient='records'), 1000)

transaction_ids = list()
for ongoing_data_chunck in ongoing_data_chuncks:  
    transaction_id = sw.transaction.log_records(
        model_id=my_model.id,
        version_id=my_version.name,
        records=ongoing_data_chunck
    )
    transaction_ids.append(transaction_id)

```
Check the status of the logged data 
```
transaction_id = sw.transaction.get(transaction_id=transaction_ids[0]['transaction_id'])
transaction_id.get_properties()['status']
# Passed'
```
### Optional - report ongoing lables to Superwise.ai

In some cases, our system is able to gather "ground truth" labels for it's predictions.
Often, this happens later on, after the prediciton was already given.

By sending these labels to Superwise.ai, we add another important layer of data to our monitoring solution.

For the purpose of this demo, we can use the test set's labels as the ground truth, simulating a label we collected in production.

```
# Note: we provide the column names we declared in the Schema object, so that Superwise.ai will be able to interpret the data
indexes = y_test.index.map(str)
ground_truth = pd.DataFrame(data=y_test, columns=['median_house_value'])
ground_truth['record_id'] = indexes
ground_truth
```


### Report the labels to Superwise.ai


```
ground_truth_chuncks = chunks(ground_truth.to_dict(orient='records'), 1000)

transaction_ids = list()
for ground_truth_chunck in ground_truth_chuncks:  
    transaction_id = sw.transaction.log_records(
        model_id=my_model.id,
        records=ground_truth_chunck
    )
    transaction_ids.append(transaction_id)

```

Check the status of the logged data
```
transaction_id = sw.transaction.get(transaction_id=transaction_ids[0]['transaction_id'])
transaction_id.get_properties()['status']
# 'Passed'
```


### ✅ Verifying the setup
![verify_the_setup](https://github.com/layerai/examples/raw/main/integrations/Superwise/images/image2.png)

🗑️ Don't forget to delete the endpoint !

```
sm_boto3.delete_endpoint(EndpointName=predictor.endpoint)
```


## Where to go from here
To learn more about using layer, you can: 
- Join our [Slack Community ](https://bit.ly/layercommunityslack)
- Checkout [Superwise documentation](https://docs.superwise.ai/)
- Visit [Layer Examples Repo](https://github.com/layerai/examples) for more examples
- Browse [Trending Layer Projects](https://layer.ai) on our main page
- Check out [Layer Documentation](https://docs.app.layer.ai) to learn more