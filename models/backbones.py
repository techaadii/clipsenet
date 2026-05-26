import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from transformers import CLIPModel

from .layers import IBN

class BottleneckIBN(nn.Module):
    """ResNet Bottleneck with Instance-Batch Normalization"""
    expansion = 4
    def __init__(self, inplanes, planes, strides=1, downsample=None):
        super(BottleneckIBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = IBN(planes)

        self.conv2 = nn.Conv2d(planes, planes, padding=1, kernel_size=3, stride=strides, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

class ResNetIBN(torch.nn.Module):
    """Modified ResNet50 acting as the fine-grained visual feature extractor"""
    def __init__(self, config):
        super(ResNetIBN, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        layer1_layers = [
            BottleneckIBN(64, 64, strides=1, downsample=self.model.layer1[0].downsample),
            BottleneckIBN(256, 64, strides=1),
            BottleneckIBN(256, 64, strides=1)
        ]
        self.model.layer1 = nn.Sequential(*layer1_layers)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)

class SemanticExtractionModule(nn.Module):
    """Extracts semantic features using frozen OpenAI CLIP"""
    def __init__(self, config):
        super(SemanticExtractionModule, self).__init__()
        self.clip = CLIPModel.from_pretrained(config.clip_model_name)
        self.vision_model = self.clip.vision_model
        self.visual_projection = self.clip.visual_projection
        
        # Freeze CLIP completely to retain open-vocabulary alignment
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.visual_projection.parameters():
            param.requires_grad = False

    def forward(self, x):
        vision_outputs = self.vision_model(pixel_values=x, interpolate_pos_encoding=True)
        semantic_features = vision_outputs.pooler_output
        return self.visual_projection(semantic_features)