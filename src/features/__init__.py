"""特徵工程。

對應 SPEC §7 的 `src/features/`「特徵工程（純函式，可測試）」。
"""

from src.features.build import (
    CATEGORICAL,
    FeatureSet,
    build_features,
)

__all__ = ["CATEGORICAL", "FeatureSet", "build_features"]
