"""Trial container and epoch views."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trial:
    subject: int | str
    condition: str
    excerpt: int
    label: str
    sfreq: float
    t0: float
    data: np.ndarray = field(repr=False)

    @property
    def key(self) -> str:
        return f"S{self.subject}_{self.condition}{self.excerpt}"


def epoch_view(x: np.ndarray, sfreq: float, length_s: float) -> np.ndarray:
    """Non-overlapping epochs, shape (n_epochs, n_channels, n_samples_epoch); trailing remainder dropped."""
    n = int(round(length_s * sfreq))
    k = x.shape[-1] // n
    return x[..., : k * n].reshape(x.shape[0], k, n).transpose(1, 0, 2)
