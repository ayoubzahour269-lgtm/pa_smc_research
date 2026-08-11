"""Fixtures partagees : construction de marches synthetiques deterministes.

Aucun test unitaire ne depend d'une donnee de marche reelle. C'est ce qui permet a la
CI de tourner alors qu'elle n'a acces a aucun fournisseur de donnees.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from alphalab.backtest.engine import SymbolData

Bar = tuple[float, float, float, float]  # open, high, low, close


def make_frame(
    bars: Sequence[Bar],
    *,
    start: str = "2020-01-06 00:00",
    freq: str = "h",
    volume: float = 1.0,
) -> pd.DataFrame:
    """Trame OHLCV valide a partir de barres explicites."""
    index = pd.date_range(start, periods=len(bars), freq=freq, tz="UTC", name="timestamp")
    arr = np.asarray(bars, dtype=float)
    return pd.DataFrame(
        {
            "open": arr[:, 0],
            "high": arr[:, 1],
            "low": arr[:, 2],
            "close": arr[:, 3],
            "volume": volume,
        },
        index=index,
    )


def flat_bars(n: int, price: float = 100.0, wick: float = 0.1) -> list[Bar]:
    """`n` barres plates : ni stop ni objectif ne peuvent etre touches."""
    return [(price, price + wick, price - wick, price)] * n


def make_market(
    frames: dict[str, pd.DataFrame], cost: float | dict[str, float] = 0.0
) -> dict[str, SymbolData]:
    """Marche synthetique avec un cout constant par symbole."""
    out: dict[str, SymbolData] = {}
    for symbol, df in frames.items():
        c = cost[symbol] if isinstance(cost, dict) else cost
        out[symbol] = SymbolData.from_frame(symbol, df, np.full(len(df), float(c)))
    return out


@pytest.fixture
def flat_market() -> dict[str, SymbolData]:
    """Un symbole, 10 barres plates a 100, cout nul."""
    return make_market({"TEST": make_frame(flat_bars(10))})
