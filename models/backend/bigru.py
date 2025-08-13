# models/backend/bigru.py
import torch
import torch.nn as nn
from typing import Optional


class BiGRUBackend(nn.Module):
    """
    Bi-directional GRU backend that accepts features shaped (B, T, D)
    and returns logits shaped (B, T, num_classes).
    """
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_classes: int = 31,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        # optionally project input_dim -> hidden_dim (if different)
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = None

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        returns: logits (B, T, num_classes)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected input (B, T, D), got shape {x.shape}")

        # Project if needed
        if self.input_proj is not None:
            x = self.input_proj(x)  # (B, T, hidden_dim)

        # GRU expects (B, T, H) with batch_first=True
        output, _ = self.gru(x)  # output: (B, T, hidden_dim * num_directions)

        logits = self.classifier(output)  # (B, T, num_classes)
        return logits


# quick sanity check
if __name__ == "__main__":
    B, T, D = 2, 75, 256
    backend = BiGRUBackend(input_dim=256, hidden_dim=512, num_layers=2, num_classes=31)
    dummy = torch.randn(B, T, D)
    out = backend(dummy)
    print("Backend out shape:", out.shape)  # expect (B, T, num_classes)
