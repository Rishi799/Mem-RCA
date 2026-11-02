"""Missingness-aware temporal graph model for service anomaly diagnosis.

This candidate research model jointly encodes per-service telemetry histories,
passes evidence along directed service-call edges, predicts next-step metrics,
and scores root-cause services. Forecast residuals provide anomaly scores; the
benchmark does not provide reliable service-level anomaly labels for a separate
supervised anomaly head.
"""
from __future__ import annotations

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - exercised without optional torch
    raise ImportError(
        "The temporal graph model requires PyTorch. Install a build compatible "
        "with your GPU from https://pytorch.org/get-started/locally/"
    ) from exc


class DirectedGraphAttention(nn.Module):
    """One directed graph-attention layer over a fixed service topology.

    ``edge_index[0]`` is caller/source and ``edge_index[1]`` is callee/target.
    Self-loops are added so each service can retain its own temporal evidence.
    """

    def __init__(self, hidden_size: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_size < 1 or heads < 1 or hidden_size % heads:
            raise ValueError("hidden_size must be positive and divisible by heads")
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_size = hidden_size // heads
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.source_attention = nn.Linear(hidden_size, heads, bias=False)
        self.target_attention = nn.Linear(hidden_size, heads, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, nodes: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if nodes.ndim != 3:
            raise ValueError("nodes must have shape (batch, services, hidden)")
        batch, service_count, hidden = nodes.shape
        if hidden != self.hidden_size:
            raise ValueError(f"Expected hidden size {self.hidden_size}, got {hidden}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, edges)")
        if edge_index.dtype != torch.long:
            raise ValueError("edge_index must use torch.long indices")
        if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= service_count):
            raise ValueError("edge_index contains a service index outside the node tensor")

        adjacency = torch.eye(service_count, dtype=torch.bool, device=nodes.device)
        if edge_index.numel():
            source, target = edge_index[0], edge_index[1]
            adjacency[target, source] = True

        # Logits are indexed [batch, target, source, head].
        source_logits = self.source_attention(nodes).unsqueeze(1)
        target_logits = self.target_attention(nodes).unsqueeze(2)
        logits = F.leaky_relu(source_logits + target_logits, negative_slope=0.2)
        logits = logits.masked_fill(~adjacency[None, :, :, None], torch.finfo(logits.dtype).min)
        weights = self.dropout(torch.softmax(logits, dim=2))

        values = self.value(nodes).reshape(batch, service_count, self.heads, self.head_size)
        aggregated = torch.einsum("btsh,bshd->bthd", weights, values).reshape(
            batch, service_count, hidden
        )
        return self.norm(nodes + self.dropout(F.elu(self.output(aggregated))))


class MissingnessAwareTemporalGraph(nn.Module):
    """Joint per-service LSTM encoders plus directed graph attention.

    Each service can have a different telemetry feature count. Inputs are lists
    ordered by a fixed, saved service list. Each sequence has shape
    ``(batch, time, features)``. The corresponding observation-mask tensor uses
    ``True``/1 for observed entries and ``False``/0 for missing entries. Missing
    values should be imputed using training-only statistics before this model;
    the explicit mask lets the network learn which inputs were imputed.
    """

    def __init__(
        self,
        feature_sizes: list[int],
        hidden_size: int = 64,
        num_layers: int = 1,
        attention_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not feature_sizes or any(size < 1 for size in feature_sizes):
            raise ValueError("feature_sizes must contain a positive size for every service")
        if hidden_size < 1 or num_layers < 1:
            raise ValueError("hidden_size and num_layers must be positive")
        self.feature_sizes = tuple(int(size) for size in feature_sizes)
        self.service_count = len(feature_sizes)
        self.hidden_size = hidden_size
        self.encoders = nn.ModuleList([
            nn.LSTM(
                input_size=2 * size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            for size in self.feature_sizes
        ])
        self.graph_attention = DirectedGraphAttention(hidden_size, attention_heads, dropout)
        self.forecast_heads = nn.ModuleList([
            nn.Linear(hidden_size, size) for size in self.feature_sizes
        ])
        self.root_cause_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        sequences: list[torch.Tensor],
        observation_masks: list[torch.Tensor],
        edge_index: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if len(sequences) != self.service_count or len(observation_masks) != self.service_count:
            raise ValueError("Provide one sequence and observation mask per configured service")
        encoded_nodes: list[torch.Tensor] = []
        batch_size: int | None = None
        for index, (sequence, mask, feature_size, encoder) in enumerate(zip(
            sequences, observation_masks, self.feature_sizes, self.encoders
        )):
            if sequence.ndim != 3 or sequence.shape[-1] != feature_size:
                raise ValueError(
                    f"Service index {index} sequence must have shape (batch, time, {feature_size})"
                )
            if mask.shape != sequence.shape:
                raise ValueError(f"Service index {index} observation mask must match its sequence")
            if batch_size is None:
                batch_size = sequence.shape[0]
            elif sequence.shape[0] != batch_size:
                raise ValueError("All service sequences must have the same batch size")
            model_input = torch.cat([sequence, mask.to(dtype=sequence.dtype)], dim=-1)
            _, (hidden, _) = encoder(model_input)
            encoded_nodes.append(hidden[-1])

        nodes = torch.stack(encoded_nodes, dim=1)
        contextual = self.graph_attention(nodes, edge_index)
        return {
            "forecast": [head(contextual[:, i, :]) for i, head in enumerate(self.forecast_heads)],
            "root_cause_logits": self.root_cause_head(contextual).squeeze(-1),
            "service_embeddings": contextual,
        }


def missingness_consistency_loss(
    reference: dict[str, torch.Tensor | list[torch.Tensor]],
    masked_view: dict[str, torch.Tensor | list[torch.Tensor]],
) -> torch.Tensor:
    """Encourage stable forecast and root rankings across masked views.

    The reference acts as a detached teacher. This term must be combined with
    supervised/forecast objectives and tuned using validation only; by itself it
    does not establish calibrated uncertainty or robustness.
    """
    reference_root = reference["root_cause_logits"]
    masked_root = masked_view["root_cause_logits"]
    assert isinstance(reference_root, torch.Tensor)
    assert isinstance(masked_root, torch.Tensor)

    reference_forecast = reference["forecast"]
    masked_forecast = masked_view["forecast"]
    assert isinstance(reference_forecast, list)
    assert isinstance(masked_forecast, list)
    if len(reference_forecast) != len(masked_forecast):
        raise ValueError("Reference and masked views must have the same service count")
    forecast_consistency = torch.stack([
        F.mse_loss(masked, clean.detach())
        for clean, masked in zip(reference_forecast, masked_forecast)
    ]).mean()
    root_target = torch.softmax(reference_root.detach(), dim=-1)
    root_consistency = F.kl_div(
        torch.log_softmax(masked_root, dim=-1), root_target, reduction="batchmean"
    )
    return forecast_consistency + root_consistency
