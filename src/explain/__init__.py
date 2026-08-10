"""M5 · 解釋層：把模型分數翻成「這個人為什麼在名單上」。

與 `src/evaluation/` 分開的理由：evaluation 回答「模型有多好、該投放給誰」，
這一層回答「為什麼是他」。前者的輸入是 (y, p)，後者的輸入是模型本身。
"""

from src.explain.attribution import (
    LOCAL_ACCURACY_ATOL,
    Attribution,
    assert_local_accuracy,
    attribute,
    local_accuracy_gap,
    mean_abs_attribution,
    sigmoid,
    top_contributors,
)
from src.explain.reasons import (
    FEATURES,
    HORIZON_EXPIRY,
    HORIZON_SHIFTED,
    HORIZON_SNAPSHOT,
    HORIZONS,
    FeatureMeta,
    add_reasons,
    expiry_dated_share,
    feature_group,
    meta,
    missing_metadata,
    render_reason,
    wide_reasons,
)

__all__ = [
    "FEATURES",
    "HORIZONS",
    "HORIZON_EXPIRY",
    "HORIZON_SHIFTED",
    "HORIZON_SNAPSHOT",
    "LOCAL_ACCURACY_ATOL",
    "Attribution",
    "FeatureMeta",
    "add_reasons",
    "assert_local_accuracy",
    "attribute",
    "expiry_dated_share",
    "feature_group",
    "local_accuracy_gap",
    "mean_abs_attribution",
    "meta",
    "missing_metadata",
    "render_reason",
    "sigmoid",
    "top_contributors",
    "wide_reasons",
]
