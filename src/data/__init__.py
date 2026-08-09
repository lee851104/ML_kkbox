"""資料載入與 as-of 切分。

對應 SPEC §7 的 `src/data/`「下載、驗證、as-of 切分」。
"""

from src.data.cohort import (
    COHORTS,
    FEB,
    MAR,
    CohortSpec,
    assert_asof_respected,
    assert_cutoffs_within_window,
    build_cohort,
    scan_transactions,
)

__all__ = [
    "COHORTS",
    "FEB",
    "MAR",
    "CohortSpec",
    "assert_asof_respected",
    "assert_cutoffs_within_window",
    "build_cohort",
    "scan_transactions",
]
