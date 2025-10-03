import torch.nn as nn
from torchvision import models, transforms

def get_mobilenet_v2(NUM_CLASSES, DEVICE):
    model_ft = models.mobilenet_v2(pretrained=True)

    # --- FREEZING LOGIC: Freeze first 90% of feature layers ---
    feature_layers = list(model_ft.features.children())
    total_layers = len(feature_layers)
    freeze_index = int(total_layers * 0.90)

    print(f"\nTotal feature blocks in MobileNetV2: {total_layers}")
    print(f"Freezing the first {freeze_index} blocks (approx. 90%).")

    # Iterate and set requires_grad
    for i, module in enumerate(feature_layers):
        for param in module.parameters():
            # True for last 10% (i >= freeze_index), False otherwise
            param.requires_grad = (i >= freeze_index)

    # --- CLASSIFIER HEAD SETUP ---
    num_ftrs = model_ft.classifier[1].in_features
    model_ft.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    model_ft = model_ft.to(DEVICE)

def mobilenet_v2_transform():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    return data_transforms