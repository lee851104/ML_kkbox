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


def make_synthetic_cohort() -> pl.DataFrame:
    """手刻一張迷你 cohort 表，欄位與 build_cohort 的輸出一致。

    用途是讓特徵層的測試不必依賴 34 GB 原始資料，因而能標成 nodata 在 CI 上
    執行。列的內容刻意涵蓋會出事的邊界：跨月與跨年的日期差、方案天數 0、
    實付 0（免費方案 vs 零收款兩種）、bd 離群值、不在 members_v3 的用戶、
    以及註冊日晚於 cutoff 的髒資料。

    全部寫死不用亂數 —— 測試失敗時要能一眼看出是哪一列出問題。
    """
    return pl.DataFrame(
        {
            "msno": [f"u{i}" for i in range(8)],
            "is_churn": [0, 1, 0, 1, 0, 1, 0, 1],
            # u1 的 cutoff 跨月：20170301 - 20170228 應該是 1 天而不是 73
            "cutoff": [
                20170228,
                20170301,
                20170216,
                20170228,
                20170101,
                20170228,
                20170215,
                20170228,
            ],
            "n_tx": [13, 1, 17, 2, 25, 1, 8, 3],
            "first_tx": [
                20160216,
                20170228,
                20151018,
                20170101,
                20150101,
                20170220,
                20160801,
                20170115,
            ],
            "last_tx": [
                20170216,
                20170301,
                20170218,
                20170228,
                20170101,
                20170228,
                20170215,
                20170228,
            ],
            "n_cancel_hist": [1, 0, 0, 2, 0, 1, 0, 0],
            "mean_paid": [99.0, 0.0, 99.0, 149.0, 180.0, 0.0, 149.0, 99.0],
            "last_is_cancel": [1, 0, 0, 1, 0, 1, 0, 0],
            "last_is_auto_renew": [1, 0, 1, 1, 1, 0, 1, 0],
            "last_actual_amount_paid": [99, 0, 99, 0, 180, 0, 149, 99],
            #                                 ↑ 免費方案      ↑ 零收款（定價 149）
            "last_plan_list_price": [99, 0, 99, 149, 180, 0, 149, 99],
            "last_payment_plan_days": [30, 7, 30, 30, 90, 0, 30, 30],
            #                                            ↑ 方案天數 0，除法要擋
            "last_payment_method_id": [41, 38, 41, 40, 39, 41, 37, 36],
            "city": [1, 13, 5, None, 1, 22, 4, None],
            "bd": [0, 25, -7168, None, 34, 2016, 45, None],
            #      ↑ 0   ↑ 有效  ↑ 極端負值      ↑ 極端正值
            "gender": ["male", None, "female", None, "male", None, "female", None],
            "registered_via": [7, 9, 7, None, 3, 4, 7, None],
            # u5 的註冊日晚於 cutoff（髒資料），年資應轉成 null 而非負數
            "registration_init_time": [
                20160216,
                20170101,
                20140310,
                None,
                20040326,
                20170330,
                20150612,
                None,
            ],
            "in_members": [True, True, True, False, True, True, True, False],
        },
        schema_overrides={
            "is_churn": pl.Int64,
            "n_tx": pl.UInt32,
            "city": pl.Int64,
            "bd": pl.Int64,
            "registered_via": pl.Int64,
            "registration_init_time": pl.Int64,
        },
    )
