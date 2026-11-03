"""PropagationNet: a graph neural network for time-to-impact forecasting.

For every time bin and service it outputs the hazard, the probability that the service's
anomaly onset falls in this bin given that it has not happened yet. Each service is
encoded from its static and dynamic features (memrca.propagation.dataset); two directed
graph-attention layers then pass evidence from its callers and from its callees (faults
mostly travel up the call chain, so the two directions are kept apart). Hazards over the
remaining bins give a survival curve: when the service is likely to be hit, and whether
it is hit at all within the horizon.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from memrca.models.temporal_graph import DirectedGraphAttention


class PropagationNet(nn.Module):
    def __init__(self, static_dim: int, dynamic_dim: int, hidden: int = 32, heads: int = 4,
                 dropout: float = 0.1, use_graph: bool = True) -> None:
        super().__init__()
        self.use_graph = use_graph
        self.encode = nn.Sequential(nn.Linear(static_dim + dynamic_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.from_callers = DirectedGraphAttention(hidden, heads, dropout)
        self.from_callees = DirectedGraphAttention(hidden, heads, dropout)
        self.hazard = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, static: torch.Tensor, dynamic: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """static (batch, N, Fs), dynamic (batch, N, Fd), edge_index (2, E) caller -> callee.
        Returns hazard logits (batch, N)."""
        h = self.encode(torch.cat([static, dynamic], dim=-1))
        edges = edge_index if self.use_graph else edge_index[:, :0]
        callers = self.from_callers(h, edges)          # a callee attends to its callers
        callees = self.from_callees(h, edges.flip(0))  # a caller attends to its callees
        return self.hazard(torch.cat([callers, callees], dim=-1)).squeeze(-1)


@torch.no_grad()
def forecast(model: PropagationNet, static: np.ndarray, dynamic_now: np.ndarray, edge_index: np.ndarray,
             start_bin: int, n_bins: int, elapsed_column: int) -> np.ndarray:
    """Survival-based forecast from ``start_bin`` using only the features observed then.

    Returns cumulative onset probabilities (n_bins - start_bin, N): entry j is the probability
    that the service has been hit by the end of bin start_bin + j.
    """
    model.eval()
    steps = n_bins - start_bin
    dyn = np.repeat(dynamic_now[None], steps, axis=0)
    dyn[:, :, elapsed_column] = (start_bin + np.arange(steps))[:, None] / n_bins
    st = np.repeat(static[None], steps, axis=0)
    logits = model(torch.from_numpy(st), torch.from_numpy(dyn.astype(np.float32)), torch.from_numpy(edge_index))
    hazard = torch.sigmoid(logits).numpy()
    return 1.0 - np.cumprod(1.0 - hazard, axis=0)
