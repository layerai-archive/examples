import argparse


import layer
from layer.decorators import model, fabric

metric_keys = [
    'train/box_loss',
    'train/obj_loss',
    'train/cls_loss',  # train loss
    'metrics/precision',
    'metrics/recall',
    'metrics/mAP_0.5',
    'metrics/mAP_0.5:0.95',  # metrics
    'val/box_loss',
    'val/obj_loss',
    'val/cls_loss',  # val loss
    'x/lr0',
    'x/lr1',
    'x/lr2']


def download_roboflow_dataset():
    from roboflow import Roboflow
    rf = Roboflow(api_key=opt.api_key)
    project = rf.workspace(opt.workspace).project(opt.project)
    dataset = project.version(1).download("yolov5")

    data_file = dataset.location + "/data.yaml"

    # Add path property
    with open(data_file, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write('path: ../' + '\r\n' + content)

    return data_file, dataset.location


def fetch_yolov5():
    import os
    os.system("git clone https://github.com/ultralytics/yolov5")
    os.system("pip install -r yolov5/requirements.txt")


def load_yolov5_module(name, path, is_class=False):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if is_class:
        return getattr(module, name)
    else:
        return module


def log_options():
    import pandas as pd
    dict = opt.__dict__.copy()
    [dict.pop(k, None) for k in ["layer_api_key", "api_key"]]
    df = pd.DataFrame({"name": dict.keys(), "value": dict.values()})
    df = df.set_index("name")
    layer.log({"parameters": df})


def on_pretrain_routine_end():
    from pathlib import Path
    layer.log({"labels": Path("./yolov5/runs/train/exp/labels.jpg")})
    layer.log({"labels_correlogram": Path("./yolov5/runs/train/exp/labels_correlogram.jpg")})


def on_fit_epoch_end(vals, epoch, best_fitness, fi):
    x = {k: float(v) for k, v in zip(metric_keys, vals)}  # dict
    print("Metrics: ", x)
    layer.log(x, epoch)


@model("detector")
@fabric("f-gpu-small")
def train():
    log_options()

    import torch

    # Load Yolov5 modules
    fetch_yolov5()
    yolov5_trainer = load_yolov5_module("train", "./yolov5/train.py")
    yolov5_Callbacks = load_yolov5_module("Callbacks", "./yolov5/utils/callbacks.py", is_class=True)

    # Download dataset
    data_file, data_location = download_roboflow_dataset()

    # Init callback
    callbacks = yolov5_Callbacks()
    callbacks.register_action("on_pretrain_routine_end", callback=on_pretrain_routine_end)
    callbacks.register_action("on_fit_epoch_end", callback=on_fit_epoch_end)

    # Parse options
    train_options = yolov5_trainer.parse_opt(True)
    options = {
        "data": data_file,
        "weights": opt.weights,
        "epochs": opt.epochs,
        "imgsz": opt.img_size,
        "batch": opt.batch,
        "workers": opt.workers,
    }
    for k, v in options.items():
        setattr(train_options, k, v)
    print("Options: ", train_options)

    # Start train
    yolov5_trainer.main(train_options, callbacks)

    # Load the best model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5/runs/train/exp/weights/best.pt')

    # Log a prediction with an image from the test set
    from pathlib import Path
    from PIL import Image
    img_path = list(Path(data_location+"/test/images").rglob("*.jpg"))[opt.test_image_index]
    img = Image.open(img_path)
    results = model(img)
    results.save(save_dir=".")

    layer.log({
        "original_img": img,
        "result": Image.open(Path(img.filename).name),
        "predictions": results.pandas().xyxy[0]
    })

    return model


opt = None


def parse_options():
    parser = argparse.ArgumentParser()

    # Layer specific options
    parser.add_argument('--layer_api_key', type=str)

    # Roboflow specific options
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--test_image_index', type=int, default=20)

    # YoloV5 specific options
    parser.add_argument('--weights', type=str, default="yolov5s.pt")
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--img_size', type=int, default=320)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--workers', type=int, default=8)


    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_options()

    if opt.layer_api_key:
        layer.login_with_api_key(opt.layer_api_key)
    else:
        layer.login()

    layer.init(project_name="yolov5-object-detection", pip_requirements_file="requirements.txt")
    layer.run([train], debug=True)