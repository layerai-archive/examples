# Image Segmentation with Tensorflow

[![Open in Layer](https://app.layer.ai/assets/badge.svg)](https://app.layer.ai/volkan/image-segmentation) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/image-segmentation/image-segmentation.ipynb) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/image-segmentation)

This is an image segmentation example based off of https://www.tensorflow.org/tutorials/images/segmentation.

## How to use

Make sure you have the latest version of the Layer SDK
```
!pip install layer-sdk -q
```

Then, you can fetch the trained model from Layer and start segmenting images

```python
import layer

image = PIL.open(...)

model = layer.get_model('volkan/image-segmentation/models/model).get_train()

pred_mask = model.predict(image)
display(image, pred_mask)
```
