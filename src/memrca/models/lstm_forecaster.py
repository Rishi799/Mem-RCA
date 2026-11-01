"""LSTM next-step forecaster used as a normal-only anomaly detector."""
from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised on systems without optional torch
    raise ImportError(
        "The Phase 2 LSTM requires PyTorch. Install a build compatible with your "
        "GPU from https://pytorch.org/get-started/locally/"
    ) from exc


class LSTMForecaster(nn.Module):
    """Predict the next service-metric vector from a fixed telemetry sequence.

    The model is trained only on pre-injection training telemetry. At inference,
    mean squared prediction error is used as the per-service anomaly score.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        output_size: int | None = None,
    ) -> None:
        super().__init__()
        if input_size < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError("input_size, hidden_size, and num_layers must be positive")
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_size, output_size or input_size)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Return the predicted next feature vector for each sequence in a batch."""
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape (batch, time, features)")
        encoded, _ = self.lstm(sequence)
        return self.output(encoded[:, -1, :])

