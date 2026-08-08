"""pytest 共用 fixture。

原始資料不進 Git，所以測試不能假設它一定存在。任何需要資料的測試在資料
缺席時要 **skip 而不是 fail** —— 一個剛 clone 完 repo 的人跑 pytest 看到
滿螢幕紅字，會以為程式壞了，實際上只是還沒下載資料。skip 才傳達正確訊息。
"""

from __future__ import annotations

import polars as pl
import pytest

from src.config import Paths, load_paths
from src.data import FEB, MAR, CohortSpec, build_cohort

# 需要掃 1.73 GB 交易檔的測試標成 slow，平常可以用
#     uv run pytest -m "not slow"
# 只跑快的那些。
SLOW = pytest.mark.slow

# 完全不碰原始資料的測試 —— 純邏輯與靜態檢查。
#
# CI 跑在 GitHub Actions 上，那裡沒有資料（資料不進 Git），所以 43 條測試裡
# 有 36 條會 skip。問題是：**一個全部 skip 的測試套件也會顯示綠燈**。
# 因此把不需資料的那些標成 nodata，CI 單獨跑一次
#     uv run pytest -m nodata
# 並要求「全過且零 skip」。這樣 CI 的綠燈才對應到「真的驗證了東西」。
#
# 判準：這個測試有沒有用到 paths / raw_files / feb_cohort / mar_cohort 任一
# fixture。有就不能標 nodata。
NODATA = pytest.mark.nodata


@pytest.fixture(scope="session")
def paths() -> Paths:
    """資料路徑。設定檔或資料缺席就 skip 整批測試。"""
    try:
        p = load_paths()
    except FileNotFoundError as e:
        pytest.skip(f"沒有路徑設定：{e}")
    if not p.raw.exists():
        pytest.skip(f"原始資料不存在（{p.raw}）。請先跑 scripts/download.py")
    return p


def _require(paths: Paths, *filenames: str) -> None:
    """確認這些檔案在 raw/ 底下，缺了就 skip。"""
    missing = [f for f in filenames if not (paths.raw / f).exists()]
    if missing:
        pytest.skip(f"缺少資料檔：{', '.join(missing)}")


@pytest.fixture(scope="session")
def raw_files(paths: Paths):
    """回傳一個「取得某個 raw CSV 的 LazyFrame」的函式，順便做存在性檢查。"""

    def get(name: str) -> pl.LazyFrame:
        _require(paths, name)
        return pl.scan_csv(paths.raw / name)

    return get


def _cohort(paths: Paths, spec: CohortSpec) -> pl.DataFrame:
    _require(paths, spec.label_file, "transactions.csv", "transactions_v2.csv", "members_v3.csv")
    # build_cohort 有快取，第一次慢、之後很快。
    return build_cohort(spec, paths, verbose=False)


@pytest.fixture(scope="session")
def feb_cohort(paths: Paths) -> pl.DataFrame:
    return _cohort(paths, FEB)


@pytest.fixture(scope="session")
def mar_cohort(paths: Paths) -> pl.DataFrame:
    return _cohort(paths, MAR)
