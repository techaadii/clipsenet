import torch
import torch.nn as nn
from .backbones import ResNetIBN, SemanticExtractionModule
from .layers import AFEMModule, FusionModule, NormalizedLinear

class CLIP_SENet(nn.Module):
    def __init__(self, config):
        super(CLIP_SENet, self).__init__()

        # Backbones
        self.backbone = ResNetIBN(config)
        self.sem_module = SemanticExtractionModule(config)
        self.sem_bn = nn.BatchNorm1d(config.clip_dim)

        # Enhancement & Fusion
        self.afem = AFEMModule(in_dim=config.clip_dim, out_dim=config.feat_dim, groups=32)
        self.fusion_module = FusionModule(config.feat_dim, config.clip_dim)

        # Learnable semantic gate to prevent CLIP from overpowering fine-grained details
        self.semantic_gate = nn.Parameter(torch.tensor([0.1]))

        # BNNeck & Classifier
        self.bottleneck = nn.BatchNorm1d(config.feat_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = NormalizedLinear(in_features=config.feat_dim, out_features=config.num_classes)

    def forward(self, images: torch.Tensor, labels=None):
        ta = self.backbone(images)
        ts = self.sem_module(images)
        ts = self.sem_bn(ts)

        ts_prime = self.afem(ts)
        fusion = self.fusion_module(ta, ts)

        # Gated addition
        t = (self.semantic_gate * ts_prime) + fusion

        feat_norm = self.bottleneck(t)
        
        if self.training:
           
            logits = self.classifier(feat_norm)
            return t, logits
        else:
            
            return feat_norm