"""反向驗證的判定邏輯 —— 純邏輯測試，不需資料。

`pairwise_stability()` 是這支腳本唯一有判斷力的地方：它決定「這個差距算不算
數」。早期版本只比排名，會把「差 0.1% 的兩個選項換位」與「真正的翻轉」判成
同一件事 —— 前者本來就該換來換去。這裡把兩種情況分別餵進去，確認它分得開。
"""

from __future__ import annotations

import pytest

from scripts.reverse_validation import NOISE_SIGMA, pairwise_stability
from tests.conftest import NODATA

# 2026-08-08 實測值（make reverse）。寫死在這裡，讓判定邏輯的改動能立刻
# 對照到「當時那批數字會被判成什麼」，而不必重跑 12 分鐘。
MEASURED = {
    "forward": {
        "CatBoost": 0.15781,
        "三者平均": 0.15796,
        "LightGBM": 0.15910,
        "XGBoost": 0.16082,
    },
    "reverse": {
        "CatBoost": 0.11597,
        "三者平均": 0.11566,
        "LightGBM": 0.11675,
        "XGBoost": 0.11758,
    },
}


def _verdict(table, pair: str) -> str:
    row = table.filter(table["對比"] == pair)
    assert row.height == 1, f"找不到對比 {pair}"
    return row["判定"][0]


@NODATA
def test_tiny_rank_flip_is_classified_as_noise():
    """CatBoost 與三者平均在兩個方向換了位，但差距只有 0.2~0.4σ。

    這種換位不構成「結論翻轉」—— 它代表兩個選項分不出高下。舊版只比排名的
    寫法會把它標成 ❌，等於用雜訊推翻自己的結論。
    """
    table = pairwise_stability(MEASURED)
    assert _verdict(table, "CatBoost vs 三者平均") == "雜訊"


@NODATA
def test_consistent_large_gap_is_trusted():
    """XGBoost 在兩個方向都明顯最差，差距都超過 1σ —— 這才是站得住的結論。"""
    table = pairwise_stability(MEASURED)
    assert _verdict(table, "CatBoost vs XGBoost") == "可信"
    assert _verdict(table, "三者平均 vs XGBoost") == "可信"


@NODATA
def test_one_sided_gap_is_flagged_as_doubtful():
    """正向 1.5σ、反向 0.9σ —— 方向一致但撐不過門檻，判為存疑而非可信。

    這一條守的是「不要因為方向一致就宣稱結論成立」。CatBoost 贏 LightGBM
    在正向看起來明確，換個方向就掉到 1σ 以下。
    """
    table = pairwise_stability(MEASURED)
    assert _verdict(table, "CatBoost vs LightGBM") == "存疑"


@NODATA
def test_sign_and_magnitude_are_both_required():
    """人造案例：方向相反且兩邊都很大 —— 這才是真正的翻轉，不得判為可信。"""
    flipped = {
        "forward": {"A": 0.100, "B": 0.100 + 3 * NOISE_SIGMA},
        "reverse": {"A": 0.100, "B": 0.100 - 3 * NOISE_SIGMA},
    }
    table = pairwise_stability(flipped)
    row = table.filter(table["對比"] == "A vs B")
    assert row["方向一致"][0] == "❌"
    assert row["判定"][0] == "存疑"


@NODATA
def test_sigma_is_configurable():
    """門檻可調。σ 放大十倍之後，原本可信的比較應該退回雜訊。

    這守的是「不要把 0.00084 這個數字寫死在判斷式裡」—— 它是量出來的，
    換一個 cohort 或換一組超參數就會變。
    """
    table = pairwise_stability(MEASURED, sigma=NOISE_SIGMA * 10)
    assert _verdict(table, "CatBoost vs XGBoost") == "雜訊"


@NODATA
def test_all_pairs_are_covered():
    """4 個模型應該產生 6 組兩兩比較，不重複也不漏。"""
    table = pairwise_stability(MEASURED)
    assert table.height == 6
    assert table["對比"].n_unique() == 6


@NODATA
def test_rejects_mismatched_model_sets():
    """兩個方向的模型清單不一致時要直接失敗，不能默默少比幾組。"""
    broken = {
        "forward": {"A": 0.1, "B": 0.2},
        "reverse": {"A": 0.1},
    }
    with pytest.raises(KeyError):
        pairwise_stability(broken)
