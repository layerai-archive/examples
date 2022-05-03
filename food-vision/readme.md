# Food vision with Layer

[![Open in Layer](https://development.layer.co/assets/badge.svg)](https://app.layer.ai/layer/image-classification) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/food-vision/food-vision.ipynb) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/food-vision)

In this project, we train a model to predict the type of food from food images.

## How to use
Make sure you have the latest version of Layer:


```
pip install layer
```
Then, you can fetch the model and predict on new food images.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KTZ3h_4OefZrQggURfr_eJClZlXp4V6g?usp=sharing)
```python
import layer 
import wget # pip install wget
import tarfile
wget.download("http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz")
food_tar = tarfile.open('food-101.tar.gz')
food_tar.extractall('.') 
food_tar.close()
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import os

image_model = layer.get_model('layer/image-classification/models/food-vision').get_train()
!wget --no-check-certificate \
    https://upload.wikimedia.org/wikipedia/commons/b/b1/Buttermilk_Beignets_%284515741642%29.jpg \
    -O /tmp/Buttermilk_Beignets_.jpg
test_image = image.load_img('/tmp/Buttermilk_Beignets_.jpg', target_size=(200, 200))
test_image = image.img_to_array(test_image)
test_image = test_image / 255.0
test_image = np.expand_dims(test_image, axis=0)
prediction = image_model.predict(test_image)
scores = tf.nn.softmax(prediction[0])
scores = scores.numpy()
base_dir = 'food-101/images'
class_names = os.listdir(base_dir)
f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } percent confidence." 
# > 'sashimi with a 1.3 percent confidence.'
```
## Dataset
The food dataset is obtained from http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz.
## Model
We train a convolution neural network model using TensorFlow to predict the type of food.

https://app.layer.ai/layer/image-classification/models/food-vision https://app.layer.ai/layer/image-classification/models/food-vision#Sample-image https://app.layer.ai/layer/image-classification/models/food-vision?tab=logs&view=9efe73fc-bd13-4945-affd-bc1c96e6efd0#Metrics https://app.layer.ai/layer/image-classification/models/food-vision#Loss-plot https://app.layer.ai/layer/image-classification/models/food-vision#Accuracy-on-test-dataset https://app.layer.ai/layer/image-classification/models/food-vision#Accuracy-plot

## Acknowledgements
The Food-101 dataset is obtained from the user [dansbecker on Kaggle](https://www.kaggle.com/datasets/dansbecker/food-101). 