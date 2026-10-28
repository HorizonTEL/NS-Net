import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)

        batch_size = features.shape[0]

        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        mask_self = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        similarity_matrix.masked_fill_(mask_self, -float('inf'))

        labels_matrix = labels.expand(batch_size, batch_size)
        labels_matrix_T = labels_matrix.T
        mask_positives = labels_matrix == labels_matrix_T
        mask_positives.masked_fill_(mask_self, False)

        loss = 0.0
        for i in range(batch_size):
            if not mask_positives[i].any():
                continue

            positive_similarities = similarity_matrix[i][mask_positives[i]]

            logits = similarity_matrix[i]
            log_prob = positive_similarities - torch.logsumexp(logits, dim=0)

            loss -= log_prob.sum() / mask_positives[i].sum()

        return loss / batch_size
