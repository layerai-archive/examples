# Image classification with Layer

[![Open in Layer](https://development.layer.co/assets/badge.svg)](https://development.layer.co/layer/cnn) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/image-classification/image_classification_with_CNNs.ipynb) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/image-classification)

## How to use

Make sure you have the latest version of Layer-SDK
```
!pip install layer-sdk -q
```

You can fetch the trained model and start making predictions from it right away. 
```python
import layer
from PIL import Image
import numpy as np
from keras.preprocessing import image
model = layer.get_model('layer/derrick-cnn/models/cnn').get_train()
!wget --no-check-certificate \
    https://upload.wikimedia.org/wikipedia/commons/c/c7/Tabby_cat_with_blue_eyes-3336579.jpg \
    -O /tmp/cat.jpg
test_image = image.load_img('/tmp/cat.jpg', target_size=(200, 200))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
prediction = model.predict(test_image)
if prediction[0][0]>0.5:
  print(" is a dog")
else:
   print(" is a cat")
# >  is a cat
```
![CAT IMAGE](https://upload.wikimedia.org/wikipedia/commons/c/c7/Tabby_cat_with_blue_eyes-3336579.jpg)
## Dataset
In this project we use the [cat and dogs dataset](https://www.kaggle.com/c/dogs-vs-cats) from Kaggle to train an image classifaction deep learning model. 
## Model 
We train a Convolutional Neural Network from scratch to predct whether a given image is a cat or a dog. 
A Convolutional Neural Network is a special class of neural networks that are built with the ability to extract unique features from image data. 
For instance, they are used in face detection and recognition because they can identify complex features in image data. 

Here's the model definition:
```python
model = Sequential([
    Conv2D(filters=32,kernel_size=(3,3),  input_shape = (200, 200, 3),activation='relu'),
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
    Dense(1, activation='sigmoid')])
```
In Keras, a Convolutional Neural Network is defined using the Conv2D layer.
The Conv2D layer expects: 
- The number of filters to be applied, in this case, 32.
- The size of the kernel, in this case, 3 by 3.
- The size of the input images. 200 by 200 is the size of the image and 3 indicates that it’s a colored image. 
- The activation function; usually ReLu. 

Check out the model on Layer on the link below:

https://development.layer.co/layer/derrick-cnn/models/cnn