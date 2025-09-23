"""
Classical model training utilities for qLLM experiments.
"""

import torch
import torch.nn as nn
from setfit import SetFitModel

# the `ModelWrapper` class provides a unified interface for
## handling tokenization
## forward passes with sentence transformer models
# here, we can work with different model architectures seamlessly


def load_model(args, device):
    """Load the pre-trained model"""
    if args.verbose:
        print(f"\nLoading pre-trained model: {args.model_name}")

    model = SetFitModel.from_pretrained(args.model_name)
    sentence_transformer = model.model_body

    # Move model to device
    model = model.to(device)
    sentence_transformer = sentence_transformer.to(device)

    if args.verbose:
        print(f"Model loaded: {type(sentence_transformer).__name__}")
        # print(f"Embedding dimension: {args.embedding_dim}")
        print(f"Model moved to device: {device}")

    return model, sentence_transformer


## SupConLoss ##


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(
        self, model, temperature=0.07, contrast_mode="all", base_temperature=0.07
    ):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, sentence_features, labels=None, mask=None):
        """Computes loss for model."""
        # Au lieu d'utiliser encode() qui peut détacher le graphe de calcul,
        # utilisons directement le modèle pour générer les embeddings
        # Tokenize the inputs
        tokenized_inputs = self.model.tokenize(sentence_features[0])

        # Si le modèle est sur un device particulier, déplacer les inputs sur ce device
        device = next(self.model.parameters()).device
        tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}

        # Forward pass avec le modèle
        outputs = self.model(tokenized_inputs)

        # Récupérer les embeddings
        if isinstance(outputs, dict) and "sentence_embedding" in outputs:
            features = outputs["sentence_embedding"]
        else:
            # Si le modèle renvoie un format différent, adaptez ici
            features = outputs  # Ou une autre méthode pour extraire les embeddings

        # Normalize embeddings
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        # Add n_views dimension
        features = torch.unsqueeze(features, 1)
        device = features.device

        # Le reste du code reste inchangé
        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
