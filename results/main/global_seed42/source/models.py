"""DeepCF and explicit feature-coupling models for aligned user-item pairs."""
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from torch import Tensor, nn

from .data import DataBundle


_VARIANTS: Final = ("deepcf", "local", "global", "coupled_local", "coupled")


@dataclass(frozen=True)
class ModelConfig:
    variant: str = "coupled"
    id_dim: int = 32
    side_dim: int = 8
    hidden_dim: int = 64
    channels: int = 8
    dropout: float = 0.5

    def __post_init__(self):
        if self.variant not in _VARIANTS:
            raise ValueError(f"Unknown model variant {self.variant!r}; expected one of {_VARIANTS}")
        for name in ("id_dim", "side_dim", "hidden_dim", "channels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.variant in ("local", "coupled_local", "coupled") and self.side_dim < 6:
            raise ValueError("side_dim must be at least 6 for a valid 3x3 convolution and 2x2 pool")
        if not isinstance(self.dropout, (float, int)) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class _ScoringModel(nn.Module):
    """Shared side-feature path and variant-specific representation fusion."""

    def __init__(self, data: DataBundle, config: ModelConfig):
        super().__init__()
        self.config = config
        self.use_deepcf = config.variant in ("deepcf", "coupled_local", "coupled")
        self.use_local = config.variant in ("local", "coupled_local", "coupled")
        self.use_global = config.variant in ("global", "coupled")
        self.use_side_features = self.use_local or self.use_global

        if self.use_side_features:
            user_features = np.asarray(data.user_features)
            item_features = np.asarray(data.item_features)
            self.register_buffer(
                "user_features",
                torch.as_tensor(user_features, dtype=torch.long),
            )
            self.register_buffer(
                "item_features",
                torch.as_tensor(item_features, dtype=torch.float32),
            )
            self.user_side_embeddings = nn.ModuleList(
                nn.Embedding(cardinality, config.side_dim)
                for cardinality in data.cardinalities
            )
            self.user_projection = nn.Linear(4 * config.side_dim, config.side_dim)
            self.item_projection = nn.Linear(item_features.shape[1], config.side_dim)
        else:
            self.user_features = None
            self.item_features = None
            self.user_side_embeddings = None
            self.user_projection = None
            self.item_projection = None

        representation_widths = []
        if self.use_deepcf:
            self.user_id_embedding = nn.Embedding(data.n_users, config.id_dim)
            self.item_id_embedding = nn.Embedding(data.n_items, config.id_dim)
            self.deepcf_projection = nn.Sequential(
                nn.Linear(config.id_dim, config.hidden_dim),
                nn.BatchNorm1d(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )
            representation_widths.append(config.hidden_dim)
        else:
            self.user_id_embedding = None
            self.item_id_embedding = None
            self.deepcf_projection = None

        if self.use_local:
            pooled_side_dim = (config.side_dim - 2) // 2
            self.local_conv = nn.Conv2d(1, config.channels, kernel_size=3)
            self.local_pool = nn.MaxPool2d(kernel_size=2)
            representation_widths.append(config.channels * pooled_side_dim**2)
        else:
            self.local_conv = None
            self.local_pool = None

        if self.use_global:
            representation_widths.append(config.side_dim**2)

        if config.variant == "deepcf":
            self.fusion = None
            self.output = nn.Linear(config.hidden_dim, 1)
        else:
            self.fusion = nn.Sequential(
                nn.Linear(sum(representation_widths), config.hidden_dim),
                nn.BatchNorm1d(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )
            self.output = nn.Linear(config.hidden_dim, 1)

        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # Conv2d keeps PyTorch's standard Kaiming initialization.

    def _validate_inputs(self, users: Tensor, items: Tensor):
        if not isinstance(users, Tensor) or not isinstance(items, Tensor):
            raise TypeError("users and items must be torch.Tensor values")
        if users.dtype != torch.int64 or items.dtype != torch.int64:
            raise TypeError("users and items must have dtype torch.int64")
        if users.ndim != 1 or items.ndim != 1 or users.shape != items.shape:
            raise ValueError("users and items must be aligned one-dimensional tensors")
        if users.device != items.device:
            raise ValueError("users and items must be on the same device")
        if self.use_side_features and users.device != self.user_features.device:
            raise ValueError("model and input tensors must be on the same device")
        if self.use_deepcf and users.device != self.user_id_embedding.weight.device:
            raise ValueError("model and input tensors must be on the same device")
        if users.numel():
            if users.min().item() < 0 or users.max().item() >= self.user_count:
                raise IndexError("user ID is outside the model's user table")
            if items.min().item() < 0 or items.max().item() >= self.item_count:
                raise IndexError("item ID is outside the model's item table")

    @property
    def user_count(self):
        if self.use_side_features:
            return self.user_features.shape[0]
        return self.user_id_embedding.num_embeddings

    @property
    def item_count(self):
        if self.use_side_features:
            return self.item_features.shape[0]
        return self.item_id_embedding.num_embeddings

    def _coupling_matrix(self, users: Tensor, items: Tensor) -> Tensor:
        user_fields = self.user_features[users]
        embedded_fields = [
            embedding(user_fields[:, field])
            for field, embedding in enumerate(self.user_side_embeddings)
        ]
        user_side = self.user_projection(torch.cat(embedded_fields, dim=1))
        item_side = self.item_projection(self.item_features[items])
        return user_side.unsqueeze(2) * item_side.unsqueeze(1)

    def forward(self, users: Tensor, items: Tensor) -> Tensor:
        self._validate_inputs(users, items)
        representations = []
        if self.use_deepcf:
            interaction = self.user_id_embedding(users) * self.item_id_embedding(items)
            deepcf = self.deepcf_projection(interaction)
            if self.config.variant == "deepcf":
                return self.output(deepcf).squeeze(1)
            representations.append(deepcf)

        if self.use_side_features:
            coupling = self._coupling_matrix(users, items)
            if self.use_local:
                local = self.local_pool(self.local_conv(coupling.unsqueeze(1)).relu())
                representations.append(local.flatten(start_dim=1))
            if self.use_global:
                representations.append(coupling.flatten(start_dim=1))

        fused = self.fusion(torch.cat(representations, dim=1))
        return self.output(fused).squeeze(1)


def _validate_data(data: DataBundle):
    if not isinstance(data, DataBundle):
        raise TypeError("data must be a DataBundle")
    user_features = np.asarray(data.user_features)
    item_features = np.asarray(data.item_features)
    if data.n_users < 1 or data.n_items < 1:
        raise ValueError("data must contain at least one user and one item")
    if user_features.shape != (data.n_users, 4) or user_features.dtype != np.int64:
        raise ValueError("user_features must be an int64 array with shape [n_users, 4]")
    if item_features.shape != (data.n_items, 18) or item_features.dtype != np.float32:
        raise ValueError("item_features must be a float32 array with shape [n_items, 18]")
    cardinalities = data.cardinalities
    if len(cardinalities) != 4 or any(cardinality < 1 for cardinality in cardinalities):
        raise ValueError("data must provide four non-empty user category vocabularies")
    for field, cardinality in enumerate(cardinalities):
        values = user_features[:, field]
        if np.any(values < 0) or np.any(values >= cardinality):
            raise ValueError(f"user feature field {field} contains an invalid category code")
    if not np.isfinite(item_features).all():
        raise ValueError("item_features must contain only finite values")


def build_model(data: DataBundle, config: ModelConfig) -> nn.Module:
    """Construct only the branches required by ``config.variant``."""
    if not isinstance(config, ModelConfig):
        raise TypeError("config must be a ModelConfig")
    _validate_data(data)
    return _ScoringModel(data, config)
