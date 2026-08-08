"""特徵工程。

對應 SPEC §7 的 `src/features/`「特徵工程（純函式，可測試）」。
"""

from src.features.build import (
    CATEGORICAL,
    FeatureSet,
    build_features,
)
from src.features.logs import (
    LOG_WINDOWS,
    assert_logs_within_cutoff,
    build_log_features,
    narrow_logs,
    window_bounds,
)

__all__ = [
    "CATEGORICAL",
    "LOG_WINDOWS",
    "FeatureSet",
    "assert_logs_within_cutoff",
    "build_features",
    "build_log_features",
    "narrow_logs",
    "window_bounds",
]
