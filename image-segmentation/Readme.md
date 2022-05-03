# Image Segmentation with Layer

[![Open in Layer](https://development.layer.co/assets/badge.svg)](https://app.layer.ai/layer/image-segmentation) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/image-segmentation/segmentation.ipynb) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/image-segmentation)

In this project, we are going to focus on image segmentation with a modified [U-net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/). 
This project has been created out of the [Image Segmentation tutorial](https://www.tensorflow.org/tutorials/images/segmentation) of Tensorflow.

## How to use

Make sure you have the latest version of Layer:
```
!pip install layer -q
```

Then, you can fetch the model to easily start image segmentation

```python
from PIL import Image
import tensorflow as tf
import requests
import matplotlib.pyplot as plt
from tensorflow.keras.utils import array_to_img

import layer
my_model = layer.get_model('layer/image-segmentation/models/mask_predictor:3.1').get_train()

def display(display_list):
    plt.figure(figsize=(10, 10))
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.imshow(array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


url = "https://previews.123rf.com/images/leonidp/leonidp1910/leonidp191000007/131898801-%C3%B0%C2%A1ouple-of-two-dogs-running-on-the-beach.jpg"
image = Image.open(requests.get(url, stream=True).raw).resize((128,128))

image_array = tf.keras.preprocessing.image.img_to_array(image)
image_array = tf.cast(image_array, tf.float32) / 255.0
pred_mask_array = my_model.predict(image_array[tf.newaxis, ...])

pred_mask_array = tf.argmax(pred_mask_array, axis=-1)
pred_mask_array = pred_mask_array[..., tf.newaxis][0]

pred_mask = array_to_img(pred_mask_array)
segmented_image = image.copy()
segmented_image.paste(pred_mask, mask=pred_mask)

display([image, pred_mask_array, segmented_image])
```

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HWMWjJukZpmtYWle87xsEOxoqXxBJaKd?usp=sharing)

## Model
We trained a modified [U-net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) model with different `Epochs` to experiment the output. 
Here our results for 10 and 20 epochs:

https://app.layer.ai/layer/image-segmentation/models/mask_predictor?w=2.1#predicted_mask


Here you can find the model:

https://app.layer.ai/layer/image-segmentation/models/mask_predictor?v=3.1


## Citation Information

```
@InProceedings{parkhi12a,
  author       = "Parkhi, O. M. and Vedaldi, A. and Zisserman, A. and Jawahar, C.~V.",
  title        = "Cats and Dogs",
  booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
  year         = "2012",
}
```
