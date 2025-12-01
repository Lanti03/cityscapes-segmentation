import torch
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101

def get_model(num_classes, backbone='resnet50', pretrained=True):
    if backbone == 'resnet50':
        model = deeplabv3_resnet50(pretrained=pretrained)
    elif backbone == 'resnet101':
        model = deeplabv3_resnet101(pretrained=pretrained)
    else:
        raise ValueError(f"Backbone {backbone} not supported.")
    
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    if model.aux_classifier is not None:
        model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
    return model
