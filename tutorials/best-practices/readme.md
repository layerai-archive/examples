# Layer best practices

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/tutorials/best-practices/best_practices.ipynb) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/tutorials/best-practices)

In this article, we will look at how get the most out of [Layer](www.layer.ai).

## Dataset saving
Often, you need to process some data and save it so as not to repeat the preprocessing steps. When working with datasets in Layer, we recommend that you: 
- Import the packages needed to preprocess the data in the dataset function. 
- Download or load the dataset in the dataset function.
- Upload your datasets to Layer, it will make using them in subsequent runs faster.  



Downloading the dataset outside the dataset function means that this data will be uploaded to Layer when you run the function. Writing the download code in the dataset function ensures that the data is downloaded directly in the container where the function is running on Layer infra. This will save you a lot of time especially when dealing with large dataset downloads. 

Expensive preprocessing steps should also be written inside the dataset function for the same reason. 

For example, the code below can be refactored with the above information in mind. 

`pip install layer -qqq`

```python
import wget 
import pandas as pd
wget.download(url)
pd.read_csv(large_downloaded_data)

```

Reading this data as a [Pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) inside the dataset function means that the data will have to be uploaded to Layer. The alternative is to use Layer resources to download this dataset. Here's how this would look like in practice: 
```python
import layer 
from layer.decorators import dataset, pip_requirements
layer.login()
layer.init("project_name")
@dataset("dataset_name")
@pip_requirements(packages=["wget"])
def save_data():
  import wget 
  import pandas as pd
  wget.download(url)
  df = pd.read_csv(large_downloaded_data)
  return df
layer.run([save_data])

```
Passing the `save_data` function to Layer means that all the instructions inside this function will be executed on Layer infra. However, if you have downloaded any large files outside this function, Layer will first pickle them and upload them. You can save some precious time by writing the download instuctions in the dataset function so that the download happens directly on Layer infra.

## Model training
In some cases, you will define large models or need to use large pre-trained models. We recommend that you write the model defiition inside the `train` function. The reasoning similar to the one we just mentioned in the dataset section above. For example, when building deep learning models we recommend that you write the instructions to download images in the training fuction. Doing otherwise means that you will have to endure longer waiting time as the images are uploaded. Writing the download instructions in the train function downloads the images on the Layer infra and they are ready to use immediately. Here is a code snippet showing how you might download some images and extract them on Layer infra. 
```python
@pip_requirements(packages=["wget"])
@fabric("f-gpu-small")
@model(name="model_name")
def train():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping
    import os
    import matplotlib.pyplot as plt 
    import pandas as pd
    import tarfile
    import wget
    wget.download(url)
    food_tar = tarfile.open('data.tar.gz')
    food_tar.extractall('.') 
    food_tar.close()
    
    base_dir = '...'
    class_names = os.listdir(base_dir)
    train_datagen = ImageDataGenerator(...)
    validation_gen = ImageDataGenerator(rescale=1./255,validation_split=0.2)
    image_size = (200, 200)
    training_set = train_datagen.flow_from_directory(...)
    validation_set = validation_gen.flow_from_directory(...)
    model =......
    model.compile()
    epochs=10
    history = model.fit(..)
    metrics_df = pd.DataFrame(history.history)
    layer.log({"Metrics":metrics_df})
    loss, accuracy = model.evaluate(..)
    layer.log({"Accuracy on test dataset":accuracy})
    metrics_df[["loss","val_loss"]].plot()
    layer.log({"Loss plot":plt.gcf()})
    metrics_df[["categorical_accuracy","val_categorical_accuracy"]].plot()
    layer.log({"Accuracy plot":plt.gcf()})
    return model
  layer.run([train])

```
## Declare dependencies 
It is good practice to declare dependencies when building entites that depend on other Layer entities. This enables Layer to optimize your pipeline. You can declare dependencies for models and datasets. 


```python
from layer import Dataset, Model

#MODEL DECORATOR WITH DEPENDENCIES
@model("clustering_model",dependencies=[Dataset("product_ids_and_vectors")])

#DATASET DECORATOR WITH DEPENDENCIES
@dataset("final_product_clusters", dependencies=[Model("clustering_model"), Dataset("product_ids_and_vectors")])


```


## Pip requirements 
[Layer fabrics](https://docs.app.layer.ai/docs/reference/fabrics) are pre-installed with common data science packages 
to make your development work faster. Check the versions of these [packages](https://docs.app.layer.ai/docs/reference/fabrics#preinstalled-libraries) 
to make sure that your project uses those versions. However, if the package version are different, we recommend that you declare the
exact version to prevent any errors. This can be done using the [pip_requirements decorator](https://docs.app.layer.ai/docs/sdk-library/pip-requirements-decorator) as shown below



```python
@pip_requirements(packages=["pandas==1.3.5","Keras==2.6.0","scikit-learn==1.0.2"])
@model(name="model_name")
def train():
    pass
```
This can be done for datasets as well:
```python
@pip_requirements(packages=["pandas==1.3.5","Keras==2.6.0","scikit-learn==1.0.2"])
@dataset(name="dataset_name")
def save_data():
    pass
```
## Where to go from here
To learn more about using layer, you can: 
- Join our [Slack Community ](https://bit.ly/layercommunityslack)
- Visit [Layer Examples Repo](https://github.com/layerai/examples) for more examples
- Browse [Trending Layer Projects](https://layer.ai) on our mainpage
- Check out [Layer Documentation](https://docs.app.layer.ai) to learn more