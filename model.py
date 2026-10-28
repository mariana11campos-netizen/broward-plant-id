"""AlexNet fine-tuned to name Broward County plants.

AlexNet comes pretrained on ImageNet (1.2 million everyday photos), so its
convolutional layers already detect edges, veins, leaf shapes and textures. We
keep those layers frozen and retrain only the fully connected classifier (plus,
optionally, the last convolutional layer) to tell our species apart. That is
what makes a few hundred photos per species enough.
"""
import torch
import torch.nn as nn
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    # Sun vs. shade vs. cloudy. Hue is left alone: flower and berry color are real clues.
    transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def build_alexnet(num_classes, pretrained=True, unfreeze_last_conv=False):
    weights = models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.alexnet(weights=weights)
    for p in net.features.parameters():
        p.requires_grad = False
    if unfreeze_last_conv:
        for p in net.features[10].parameters():  # conv5
            p.requires_grad = True
    net.classifier[6] = nn.Linear(4096, num_classes)
    return net


class PlantClassifier:
    """Loads models/alexnet_plants.pt and names the plant in one PIL image."""

    def __init__(self, weights_path):
        ckpt = torch.load(weights_path, map_location="cpu")
        self.classes = ckpt["classes"]
        self.net = build_alexnet(len(self.classes), pretrained=False)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()

    def predict(self, img, top_k=3):
        """Returns [(class_id, probability), ...] best first."""
        x = eval_tf(img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.net(x), dim=1)[0]
        top = probs.topk(min(top_k, len(self.classes)))
        return [(self.classes[i], float(p)) for p, i in zip(top.values, top.indices)]
