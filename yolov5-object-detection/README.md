# COCO Instance Segmentation

[![Open in Layer](https://app.layer.ai/assets/badge.svg)](https://app.layer.ai/layer/yolov5-object-detection/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/yolov5-object-detection/notebooks/demo.ipynb) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/yolov5-object-detection)

![YoloV5 Object Detection](https://raw.githubusercontent.com/layerai/examples/main/yolov5-object-detection/assets/yolov5_object_detection.png)

In this tutorial, we are going to train an **YOLOv5 Object Detection** model with a dataset from [Roboflow Universe](https://universe.roboflow.com/). 

Roboflow Universe hosts free public computer vision datasets in many popular formats (including CreateML JSON, COCO JSON, Pascal VOC XML, YOLO v3, and Tensorflow TFRecords). For this example 

## How to train a Yolov5 Object Detection model

With Layer, you can start training your **YoloV5 Object Detection** model with a [Roboflow Universe](https://universe.roboflow.com/) dataset easily.

First make sure you have the latest Layer SDK:
```shell
pip install layer -U
```

Then clone the [Layer Examples](https://github.com/layerai/examples) which contains this project
```shell
git clone https://github.com/layerai/examples
cd layerai/examples/yolov5-objection-detection
```

And in your terminal just run the following command, that's all! Easy right!
```shell
python main.py --api_key=[YOUR_ROBOFLOW_API_KEY] --workspace=[ROBOFLOW_WORKSPACE] --project=[ROBOFLOW_PROJECT]
```

Layer will fetch your Roboflow dataset and start training with YoloV5. You will be able to find all metrics and sample predictions
in your Layer Project.

You can also pass hyper parameters to your YoloV5 train with. Run the following command for all parameters you can pass:
```shell
python main.py --help
```


## How to make predictions

Once you train you model, you can load and use the **YoloV5** model easily. 

First make sure you have the required libraries.

```
!pip install layer -q
!git clone https://github.com/ultralytics/yolov5
!pip install -r yolov5/requirements.txt
```

Then, you can fetch the model from your project and make predictions. Here is a sample code on using the model we have 
trained in this project.

```python
import layer

my_model = layer.get_model("layer/yolov5-object-detection/models/detector").get_train()

img = "https://www.mammoet.com/siteassets/equipment/transport/heavy-duty-rail-cars/Heavy-duty-rail-cars.jpg"
results = my_model(img)

results.print()

# image 1/1: 1728x2304 1 engine
```

Here is a sample prediction logged during the training of this model:

https://app.layer.ai/layer/yolov5-object-detection/models/detector?v=2.2#result

## Roboflow Universe Object Detection Dataset

We used the [RailsCars Image Dataset](https://universe.roboflow.com/new-workspace-w5mg3/railcarsv2_3) from Roboflow Universe for this project. We have exported the dataset in Yolo format and trained the Yolov5 Object Detection model.

The dataset have 2376 images labeled with 18 categories.

## Experiments

We started our **machine learning experiments** with multiple parameters. 
Here is a comparison of the parameters in our best performing models:

https://app.layer.ai/layer/yolov5-object-detection/models/detector?w=2.1&w=1.1#parameters

Mostly we have experimented with number of `epochs` and `image size` parameters. Our target metric was `mAP_0.5`.
As you can see in the **metric comparison chart** all experiments was very close. 
https://app.layer.ai/layer/yolov5-object-detection/models/detector?w=2.1&w=1.1#metrics/mAP_0.5

Our winning model was trained in 10 `epochs` with 640 `img size`. Here you can see the detection results in different runs.
https://app.layer.ai/layer/yolov5-object-detection/models/detector?w=2.1&w=1.1#result


## References
- https://universe.roboflow.com/
- https://github.com/ultralytics/yolov5