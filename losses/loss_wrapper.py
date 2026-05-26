import torch
import torch.nn as nn
import torch.nn.functional as F

class LossWrapper(nn.Module):
    """Computes CE Loss (for identity) and Supervised Contrastive Loss (for clustering)"""
    def __init__(self, num_classes, epsilon=0.1, temperature=0.07):
        super(LossWrapper, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=epsilon)
        self.temperature = temperature

    def supercon_loss(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        logit_masks = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logit_masks
        logits = similarity_matrix / self.temperature

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * logit_masks
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mask_pos_pairs = mask.sum(1)
        # Avoid division by zero
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        loss = -mean_log_prob_pos.mean()

        return loss
    
    def forward(self, cls_score, features, targets):
        ce_loss = self.ce_loss(cls_score, targets)
        sc_loss = self.supercon_loss(features, targets)
        total_loss = ce_loss + sc_loss
        return total_loss, ce_loss, sc_loss