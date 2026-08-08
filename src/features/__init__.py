"""特徵工程。

對應 SPEC §7 的 `src/features/`「特徵工程（純函式，可測試）」。
"""

from src.features.build import (
    CATEGORICAL,
    FeatureSet,
    build_features,
)
from src.features.encoding import (
    TargetEncoder,
    assert_encoding_is_oof,
    fit_target_encoder,
    oof_target_encode,
)
from src.features.logs import (
    LOG_GROUPS,
    LOG_WINDOWS,
    assert_logs_within_cutoff,
    build_log_features,
    log_feature_group,
    narrow_logs,
    window_bounds,
)

__all__ = [
    "CATEGORICAL",
    "LOG_GROUPS",
    "LOG_WINDOWS",
    "FeatureSet",
    "TargetEncoder",
    "assert_encoding_is_oof",
    "assert_logs_within_cutoff",
    "build_features",
    "build_log_features",
    "fit_target_encoder",
    "log_feature_group",
    "narrow_logs",
    "oof_target_encode",
    "window_bounds",
]
