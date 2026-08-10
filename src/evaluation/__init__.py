"""評估指標與分群回報。

SPEC §7 的結構表沒有列這一層（只列了 data / features / models / serving），
這是刻意的追加。理由：log loss 與分群回報同時被「訓練時的驗證」與「報表
產生」使用，放在 models/ 底下會讓報表程式看起來依賴模型層，實際上並沒有。
"""

from src.evaluation.calibration import (
    brier_score,
    calibration_in_the_large,
    expected_calibration_error,
    max_calibration_error,
    reliability_curve,
)
from src.evaluation.calibrator import (
    IsotonicCalibrator,
    fit_isotonic,
    tie_profile,
)
from src.evaluation.metrics import (
    EPS,
    constant_log_loss,
    log_loss,
    repeat_vs_new,
    roc_auc,
    segment_report,
)

__all__ = [
    "EPS",
    "IsotonicCalibrator",
    "brier_score",
    "calibration_in_the_large",
    "constant_log_loss",
    "expected_calibration_error",
    "fit_isotonic",
    "log_loss",
    "max_calibration_error",
    "reliability_curve",
    "repeat_vs_new",
    "roc_auc",
    "segment_report",
    "tie_profile",
]
