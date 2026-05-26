import torch
import torch.nn as nn
import torch.nn.functional as F

class IBN(nn.Module):
    def __init__(self, planes: int):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split_size = x.size(1) // 2
        split = torch.split(x, split_size, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        return torch.cat((out1, out2), 1)

class NormalizedLinear(nn.Module):
    """L2-Normalized Classifier to prevent score inflation"""
    def __init__(self, in_features, out_features, scale=30.0):
        super(NormalizedLinear, self).__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return self.scale * F.linear(x_norm, w_norm)

class FusionModule(nn.Module):
    def __init__(self, feat_dim, clip_dim):
        super(FusionModule, self).__init__()
        self.dim_reduce = nn.Sequential(
            nn.Linear(feat_dim + clip_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim)
            
        )

    def forward(self, visual_features, semantic_features):
        f1 = visual_features.view(visual_features.size(0), -1)
        f2 = semantic_features.view(semantic_features.size(0), -1)
        fused_features = torch.cat((f1, f2), dim=1)
        return self.dim_reduce(fused_features)

class AFEMModule(nn.Module):
    def __init__(self, in_dim, out_dim, groups):
        super(AFEMModule, self).__init__()
        self.groups = groups
        self.f_linear = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim)
            
        )
        self.W = nn.Parameter(torch.zeros(1, groups) + 1e-3)

    def forward(self, t_s):
        x = self.f_linear(t_s)
        batch, dim = x.shape
        channels_per_group = dim // self.groups
        
        x_grouped = x.view(batch, self.groups, channels_per_group)
        expanded_weights = self.W.view(1, self.groups, 1)
        weighted_out = (x_grouped * expanded_weights).view(batch, dim)
        
        return x + weighted_out