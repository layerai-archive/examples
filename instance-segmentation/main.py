import layer
from layer.decorators import model, fabric, resources
import engine
import transforms
import utils
import coco_eval
import coco_utils
import cloudpickle
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import json
from pycocotools import mask as pctmask
from os import path
import argparse


class CocoInstanceDataset(torch.utils.data.Dataset):

    def __init__(self, image_root, instance_dataset, class_ids, transforms=None):
        self.root = image_root
        self.transforms = transforms
        self.class_ids = class_ids
        f = open(instance_dataset)
        coco_file = json.load(f)
        self.images = coco_file["images"]
        self.annotations = coco_file["annotations"]
        self.imgs = []
        for ann in self.annotations:
            cat_id = ann["category_id"]
            img_id = ann["image_id"]
            if cat_id in class_ids and img_id not in self.imgs:
                for img in self.images:
                    if img["id"] == img_id:
                        if path.exists(os.path.join(self.root, img["file_name"])):
                            self.imgs.append(img_id)
        print(len(self), " found")

    def __getitem__(self, idx):
        # load images ad masks
        image_id = self.imgs[idx]

        image = None
        for img in self.images:
            if img["id"] == image_id:
                image = img
                break

        anns = []
        for ann in self.annotations:
            if ann["image_id"] == image_id and ann["category_id"] in self.class_ids:
                anns.append(ann)

        img_path = os.path.join(self.root, image["file_name"])
        img = Image.open(img_path).convert("RGB")

        mask = None
        labels = []
        for ann in anns:
            seg = ann["segmentation"]
            ann_id = ann["id"]
            cat_id = ann["category_id"]
            labels.append(cat_id)
            seg_mask = pctmask.decode(seg)
            seg_mask = np.where(seg_mask > 0, ann_id, seg_mask)

            if mask is None:
                mask = seg_mask
            else:
                mask = mask + seg_mask

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1]) + 1
            ymin = np.min(pos[0])
            ymax = np.max(pos[0]) + 1
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def fetch_dataset(name, version, api_key):
    from segments import SegmentsClient, SegmentsDataset
    from segments.utils import export_dataset

    client = SegmentsClient(api_key)
    release = client.get_release(name, version)
    dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['labeled', 'reviewed'])

    return export_dataset(dataset, export_format='coco-instance')


def get_instance_segmentation_model(num_classes):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(is_train):
    # converts the image, a PIL image, into a PyTorch Tensor
    train_transforms = [transforms.ToTensor()]
    if is_train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        train_transforms.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(train_transforms)


def get_datasets(img_root, coco_json, class_ids):
    import torch
    # use our dataset and defined transformations

    dataset = CocoInstanceDataset(img_root,
                                  coco_json,
                                  class_ids,
                                  get_transform(is_train=True))
    dataset_test = CocoInstanceDataset(img_root,
                                       coco_json,
                                       class_ids,
                                       get_transform(is_train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:opt.train_dataset_limit])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-opt.test_dataset_limit:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.train_batch_size, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    return data_loader, data_loader_test


def pred2image(img_as_tensor, prediction):
    from PIL import Image
    import random

    img = Image.fromarray(img_as_tensor.mul(255).permute(1, 2, 0).byte().numpy())
    index = 0
    for score in prediction[0]['scores'].cpu().numpy():
        if score > 0.7:
            marr = prediction[0]['masks'][index, 0].mul(255).byte().cpu().numpy()
            marr = np.dstack((marr, marr, marr))
            alpha = np.sum(marr, axis=-1) > (90 * 3)
            alpha = np.uint8(alpha * 150)
            marr = np.dstack((marr, alpha))
            marr[:, :, 0] = marr[:, :, 0] * random.uniform(0, 1)
            marr[:, :, 1] = marr[:, :, 1] * random.uniform(0, 1)
            marr[:, :, 2] = marr[:, :, 2] * random.uniform(0, 1)
            obj = Image.fromarray(marr.astype(np.uint8)).convert("RGBA")
            img.paste(obj, mask=obj.split()[3])
        index += 1

    return img


def log_options():
    import pandas as pd
    dict = opt.__dict__.copy()
    [dict.pop(k, None) for k in ["layer_api_key", "segments_api_key"]]
    df = pd.DataFrame({"name": dict.keys(), "value": dict.values()})
    df = df.set_index("name")
    layer.log({"parameters": df})


@model("object_detector")
@fabric("f-gpu-small")
def train():
    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    log_options()

    # load datasets
    coco_json, img_root = fetch_dataset(opt.dataset, opt.dataset_version, opt.segments_api_key)

    # test datasets
    # img_root = './images'
    # coco_json = './export_coco-instance_segments_sidewalk-imagery_v1.0.json'

    if opt.class_ids is None:
        class_ids = list(range(1, 35))
    else:
        class_ids = list(map(int, opt.class_ids.split(',')))

    data_loader, data_loader_test = get_datasets(img_root, coco_json, class_ids)

    layer.log({"train size": len(data_loader.dataset),
               "test size": len(data_loader_test.dataset),
               })

    num_classes = 35
    if opt.fine_tune:
        model = layer.get_model(opt.fine_tune).get_train()
    else:
        model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size,
                                                   gamma=opt.gamma)

    num_epochs = opt.epochs
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        engine.train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()

    # evaluate on the test dataset
    engine.evaluate(model, data_loader_test, device=device)

    # log sample prediction
    img, target = data_loader_test.dataset[20]
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
        layer.log({"prediction": pred2image(img, prediction)})

    return model


cloudpickle.register_pickle_by_value(transforms)
cloudpickle.register_pickle_by_value(utils)
cloudpickle.register_pickle_by_value(engine)
cloudpickle.register_pickle_by_value(coco_eval)
cloudpickle.register_pickle_by_value(coco_utils)

opt = None


def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--layer_api_key', type=str, required=True)
    parser.add_argument('--segments_api_key', type=str, required=True)
    parser.add_argument('--dataset', type=str, default="segments/sidewalk-imagery")
    parser.add_argument('--dataset_version', type=str, default="v1.0")
    parser.add_argument('--fine-tune', type=str, default=None)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--step_size', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--train_dataset_limit', type=int, default=800)
    parser.add_argument('--test_dataset_limit', type=int, default=50)
    parser.add_argument('--class_ids', type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_options()

    if opt.layer_api_key:
        layer.login_with_api_key(opt.layer_api_key)
    else:
        layer.login()

    layer.init(project_name="instance-segmentation", pip_requirements_file="requirements.txt")
    # layer.run([train])

    # train()
