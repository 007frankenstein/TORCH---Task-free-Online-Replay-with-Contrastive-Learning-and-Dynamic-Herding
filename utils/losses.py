"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda:0')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -1 * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    

class PCRLoss(nn.Module):

    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super(PCRLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda:0')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        elif self.contrast_mode == 'proxy':
            anchor_feature = features[:, 0]
            contrast_feature = features[:, 1]
            anchor_count = 1
            contrast_count = 1
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        # compute log_prob
        if self.contrast_mode == 'proxy':
            exp_logits = torch.exp(logits)
        else:
            mask = mask * logits_mask
            exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -1 * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class PrototypeContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(PrototypeContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: tensor of shape [bsz, n_views, dim]
            labels: tensor of shape [bsz]
        Returns:
            A scalar loss.
        """
        device = features.device
        bsz, n_views, dim = features.shape
        features = features.view(bsz * n_views, dim)
        labels = labels.repeat(n_views)  # Expand labels for multiple views

        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute class prototypes in the batch
        unique_labels = labels.unique()
        prototypes = []
        for c in unique_labels:
            mask = labels == c
            if mask.sum() > 0:
                prototype = features[mask].mean(dim=0)
                prototypes.append(prototype)
            else:
                prototypes.append(torch.zeros(dim).to(device))
        prototypes = torch.stack(prototypes, dim=0)  # [n_classes, dim]

        # Compute similarity between features and all class prototypes
        logits = torch.matmul(features, prototypes.T) / self.temperature  # [bsz * n_views, n_classes]

        # Create targets: for each feature, find the index of its true class in `unique_labels`
        target_indices = torch.zeros_like(labels)
        for idx, label in enumerate(unique_labels):
            target_indices[labels == label] = idx

        loss = F.cross_entropy(logits, target_indices.to(device))
        return loss
    


# class CeCLLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, logits: torch.Tensor, labels: torch.Tensor):
#         """
#         logits: [B, C] cosine-similarity logits
#         labels: [B]
#         """
#         B, C = logits.shape
#         device = logits.device

#         # Edge case: only one class
#         if labels.unique().numel() <= 1:
#             return logits.sum() * 0.0

#         # Numerical stabilization
#         logits = logits - logits.max(dim=1, keepdim=True)[0]

#         log_probs = F.log_softmax(logits, dim=1)

#         # Positive term
#         pos_log_prob = log_probs[torch.arange(B), labels]

#         # Negative term: mean over non-true labels
#         neg_mask = torch.ones_like(log_probs, dtype=torch.bool)
#         neg_mask[torch.arange(B), labels] = False

#         neg_log_prob = log_probs[neg_mask].view(B, -1).mean(dim=1)

#         loss = -(pos_log_prob + neg_log_prob).mean()
#         return loss
    

# class CeCLLoss(nn.Module):
#     def __init__(self, eps=1e-8):
#         super().__init__()
#         self.eps = eps

#     def forward(self, logits: torch.Tensor, labels: torch.Tensor):
#         """
#         logits: [B, C]
#         labels: [B]
#         """
#         B, C = logits.shape
#         device = logits.device

#         # Convert logits → probabilities
#         probs = F.softmax(logits, dim=1)
#         probs = probs.clamp(self.eps, 1.0 - self.eps)

#         # Positive term: log f_{y_i}(x_i)
#         pos = probs[torch.arange(B, device=device), labels]
#         pos_term = torch.log(pos)

#         # Negative term: mean over y ≠ y_i of log(1 - f_y(x_i))
#         neg_mask = torch.ones_like(probs, dtype=torch.bool)
#         neg_mask[torch.arange(B, device=device), labels] = False

#         neg_probs = probs[neg_mask].view(B, C - 1)
#         neg_term = torch.log(1.0 - neg_probs).mean(dim=1)

#         # Final loss
#         loss = -(pos_term + neg_term).mean()
#         return loss


class CeCLLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(CeCLLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, targets, prototypes):
        """
        Streamlined CeCL Loss assuming prototypes for 'targets' always exist.
        """
        # 1. Prepare Prototypes & Mappings
        # Sort keys to ensure consistent column ordering (0..C-1)
        active_classes = sorted(list(prototypes.keys()))
        
        # We still need to map Global Class IDs (e.g., 105) to Matrix Indices (e.g., 3)
        # This is fast because it's just a dictionary lookup
        class_to_idx = {cls_id: idx for idx, cls_id in enumerate(active_classes)}
        
        mapped_targets = torch.tensor(
            [class_to_idx[t.item()] for t in targets], 
            device=features.device
        )

        # 2. Compute Logits vs ALL Prototypes (Required for Denominator)
        # Stack prototypes into matrix [C_total, D]
        proto_tensor = torch.stack([prototypes[c] for c in active_classes])
        proto_tensor = proto_tensor.to(features.device)
        proto_tensor = F.normalize(proto_tensor, p=2, dim=1) 

        # [B, C_total]
        logits = torch.mm(features, proto_tensor.T) / self.temperature
        log_probs = F.log_softmax(logits, dim=1) 

        # 3. Separate Numerator (True Class) & Denominator (Negatives)
        
        # OPTIMIZATION: Use 'gather' instead of one-hot multiplication.
        # It extracts the value at the specific index for each row.
        # mapped_targets.view(-1, 1) reshapes to column vector for gather
        numerator = log_probs.gather(1, mapped_targets.view(-1, 1)).squeeze()
        
        # Denominator: Average of Negative Log-Probs
        # Sum of ALL logs minus the True log gives us the Sum of Negatives
        sum_neg_logs = log_probs.sum(dim=1) - numerator
        
        num_classes = len(active_classes)
        
        if num_classes > 1:
            avg_neg_logs = sum_neg_logs / (num_classes - 1)
        else:
            # If there is only 1 class so far, we cannot compute negatives.
            # Loss is 0 because discrimination is impossible.
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # 4. Compute Ratio (Minimize this)
        # Ratio = log(P_true) / avg(log(P_neg))
        # Add epsilon to denominator to prevent div-by-zero
        ratio = numerator / (avg_neg_logs - 1e-8)
        
        return ratio.mean()

