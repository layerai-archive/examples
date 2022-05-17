# How to integrate Layer and Ango
In this tutorial, we'll look at how to integrate [Layer](layer.ai) and [Ango](http://ango.ai/) in your machine learning projects. 

[![Open in Layer](https://development.layer.co/assets/badge.svg)](https://app.layer.ai/layer/ango-face-classification) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/integrations/Ango/ango.ipynb) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/integrations/Ango)

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
## Initialize your first Layer project
It's time to create your first Layer Project. You can find your created project at https://app.layer.ai. 

```
layer.init("face-classification")

```
## Fetch data from Ango Hub
For this illustration, we use [Ango's face classification dataset](https://ango.ai/open-dataset/). 

We can use the Ango SDK to fetch data from Ango Hub. You will need to go to [ango.ai](https://ango.ai/), create an account and then obtain 
your API key and project ID. 

Let's define a fuction to fetch the data using Ango: 
```python
from ango.sdk import SDK
import os 
import urllib.request

class Ango:
    def __init__(self, api_key, project_id=None) -> None:
        self.sdk = SDK(api_key=api_key)
        if project_id:
            self.project_id = project_id
    
    def setProject(self,project_id):
        self.project_id = project_id
    
    '''
    Gets annotations for assets within a project, streams page by page.
    params:
    items_per_page : The number of annotations fetched per page.
    annotation_status : The current stage of annotation ("Completed" OR "Todo") leave blank to fetch all

    returns:
    A List of annotations
    '''
    def getAnnotations(self, items_per_page = 100, annotation_status = None):
        remaining_tasks = 1
        page = 1
        tasks = []
        while (remaining_tasks > 0):
            response =  self.sdk.get_tasks(self.project_id, page=page, limit=items_per_page, status= annotation_status)
            tasks.extend(response['data']['tasks'])
            remaining_tasks =  response["data"]["total"] - len(tasks)
            page += 1
        return tasks

    def get_name_from_url(self, imgUrl):
      return imgUrl.split('/')[-1]

    def fetchImages(self,images, folder_path="downloaded_images/"):
      dirname = os.path.dirname(__file__)
      if (not os.path.exists(folder_path)):
          os.mkdir(os.path.join(dirname, folder_path))
      for imgUrl in images:
        img_name = self.get_name_from_url(imgUrl)
        image_path = os.path.join(dirname, folder_path, img_name)
        if os.path.isfile(image_path):
          continue
        else:
          urllib.request.urlretrieve(imgUrl, image_path)
      print("All images downloaded")

    def fetchExportLink(self):
      return self.sdk.export(self.project_id)['data']['exportPath']
#Run this block after the two credentials have been added.
#You may save the annotations in JSON, or use them programatically. 
#Note: This takes some time for larger annotations.
ango = Ango(api_key="YOUR_API_KEY",project_id="YOUR_PROJECT_ID") #Face Classification
annotations = ango.getAnnotations(annotation_status="Completed")
print(len(annotations))
      
```
### Save the data as a Pandas DataFrame
```python
import pandas as pd
import io
import base64

def get_answer(schemaId, task):
  return next((answer["answer"] for answer in task['answer']['classifications'] if answer['schemaId'] == schemaId), None)

def build():
  from PIL import Image
  import requests

  data = []
  for task in annotations[:2500]:
    img_url = task["asset"]["data"]
    img = Image.open(requests.get(img_url, stream=True).raw)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=img.format)
    image_as_string = base64.b64encode(img_byte_arr.getvalue())
    img_str = image_as_string.decode("utf-8")

    data.append([
      img_str,
      get_answer("7d4d70ea16e8e5d7ce8e721", task), # Sex
      get_answer("76e4a3dbf96926edadd5203", task), # Age
      get_answer("05e865541776c186f3e4003", task), # Hair Color
      get_answer("ff5e7ac66607ebe73810601", task), # Beard Color
      get_answer("1f5411a7bbcdba28fe30677", task), # Mustache Color
      get_answer("d0d75fc06feaa006e5c0106", task), # Eye Color
      get_answer("ec5e7cd3838fc4d5c7c6298", task), # Glasses
    ])
    # answers = task['answer']['classifications']
    # answer = next((answer for answer in task['answer']['classifications'] if answer['schemaId'] == "7d4d70ea16e8e5d7ce8e721"), None)
    # print(answer)
    task['answer']['classifications']
  
  return pd.DataFrame(data,columns=["image", "sex", "age","hair_color","beard_color","mustache_color","eye_color","glasses"])
```
### Label encode the gender column
We'll build a model to predict the gender of a person. So let's encode that column. You can do the same for the other 
columns. 
```python
# creating instance of labelencoder
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
gender = gender.assign(sex = labelencoder.fit_transform(gender["sex"]))
```
### Process the image data
```python
from tensorflow.keras.preprocessing.image import img_to_array
def load_process_images(image):
  image = image.resize((224,224))
  image_array  = img_to_array(image)
  return image_array
```
## Train the model on Layer 
To train the model on Layer, we create a function decorated with 
the [@model decorator](https://docs.app.layer.ai/docs/sdk-library/model-decorator). We also use, the [@fabric decorator](https://docs.app.layer.ai/docs/reference/fabrics#predefined-fabrics) 
to indicate that we want to train the model on layer GPUs. 
```python
@fabric("f-gpu-small")
@model("face-classification")
@fabric("f-gpu-small")
@model("face-classification")
def train():
  from tensorflow import keras
  from tensorflow.keras import Sequential
  from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,Resizing
  from tensorflow.keras.preprocessing.image import ImageDataGenerator
  from tensorflow.keras.callbacks import EarlyStopping
  import matplotlib.pyplot as plt 
  import numpy as np
  import pandas as pd
  from sklearn.preprocessing import LabelEncoder
  from sklearn.model_selection import train_test_split


  df = build()
  gender = df[['image','sex']]
  labelencoder = LabelEncoder()
  gender = gender.assign(sex = labelencoder.fit_transform(gender["sex"]))
  X = gender[['image']]
  y = gender[['sex']]
  X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42)

  for image in range(4):
    layer.log({f"Sample face-{image}": X_train['image'][image]})
  X_train = np.stack(X_train['image'].map(load_process_images))
  X_test = np.stack(X_test['image'].map(load_process_images))
    
  train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2,zoom_range=0.2, horizontal_flip=True,width_shift_range=0.1,height_shift_range=0.1)
  train_datagen.fit(X_train)
  training_data = train_datagen.flow(X_train, y_train, batch_size=32)
 
  validation_gen = ImageDataGenerator(rescale=1./255)
  testing_data = validation_gen.flow(X_test, y_test, batch_size=32)

  model = Sequential([
    Conv2D(filters=32,kernel_size=(3,3),  input_shape = (224, 224, 3),activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(filters=32,kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    Conv2D(filters=64,kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(3, activation='softmax')])

  model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics=[keras.metrics.CategoricalAccuracy()])
  callback = EarlyStopping(monitor='loss', patience=3)
  epochs=3
  history = model.fit(training_data,validation_data=testing_data, epochs=epochs,callbacks=[callback])
  metrics_df = pd.DataFrame(history.history)
  layer.log({"metrics DataFrame": metrics_df})
  loss, accuracy = model.evaluate(testing_data)
  layer.log({"Testing loss": loss})
  layer.log({"Testing accuracy": accuracy})
  print('Accuracy on test dataset:', accuracy)
  metrics_df[["loss","val_loss"]].plot()
  layer.log({"Loss plot": plt.gcf()})
  training_loss, training_accuracy = model.evaluate(training_data)
  layer.log({"Training loss": training_loss})
  layer.log({"Training accuracy": training_accuracy})
  metrics_df[["categorical_accuracy","val_categorical_accuracy"]].plot()
  layer.log({"Accuracy plot": plt.gcf()})
  return model
# Run the project on Layer Infra using remote GPUs
layer.run([train])
```
https://app.layer.ai/layer/ango-face-classification/models/face-classification?w=32.1&w=25.1&w=19.1&w=17.1&w=14.1&w=12.1#Sample-face-0 https://app.layer.ai/layer/ango-face-classification/models/face-classification?w=32.1&w=25.1&w=19.1&w=17.1&w=14.1&w=12.1#metrics-DataFrame https://app.layer.ai/layer/ango-face-classification/models/face-classification?w=32.1&w=25.1&w=19.1&w=17.1&w=14.1&w=12.1#Loss-plot https://app.layer.ai/layer/ango-face-classification/models/face-classification?w=32.1&w=25.1&w=19.1&w=17.1&w=14.1&w=13.1&w=12.1#
## Run predictions on the model
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_Pj_AhadYI-iRyMV2D0bq5_ncgvLYPn6?usp=sharing)

We now fetch the model trained on Layer and start making predictions with new faces. 
```python
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
my_model = layer.get_model('layer/ango-face-classification/models/face-classification').get_train()
!wget --no-check-certificate \
   https://storage.googleapis.com/ango-covid-dataset/ffhq-dataset/batch2/48312.png \
    -O 48312.png
test_image = image.load_img('48312.png', target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = test_image / 255.0
test_image = np.expand_dims(test_image, axis=0)
prediction = my_model.predict(test_image)
scores = tf.nn.softmax(prediction[0])
scores = scores.numpy()
class_names = ["Male","I am not sure","Female"]
f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } percent confidence." 
# > Male with a 57.42 percent confidence.
```
![Face](https://storage.googleapis.com/ango-covid-dataset/ffhq-dataset/batch2/48312.png)
## Next steps
To learn more about using layer, you can: 
- Join our [Slack Community ](https://bit.ly/layercommunityslack)
- Visit [Layer Examples Repo](https://github.com/layerai/examples) for more examples
- Browse [Trending Layer Projects](https://layer.ai) on our mainpage
- Check out [Layer Documentation](https://docs.app.layer.ai) to learn more