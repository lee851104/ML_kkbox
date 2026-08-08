"""紅線測試 —— SPEC §5 的八條紅線。

SPEC §5 開宗明義：「以下任何一條被違反，該版本作廢重做。每一條都要有對應的
**會失敗的測試**。」

「會失敗的測試」的重點在於：光證明「現在的資料通過檢查」是不夠的，因為一個
永遠回傳 True 的空檢查也會通過。必須能餵給它一份確實違規的輸入，確認它真的
會擋下來。下面 test_red_line_1 就是這樣寫的。

目前八條裡只有 4 條寫得出來（1、3、7、8）。另外 4 條依賴還不存在的程式碼 ——
為不存在的東西寫測試只能寫出假的通過。它們以 skip 保留在清單裡，每次 pytest
都會提醒還欠幾條，等對應的里程碑做到就填上。
"""

from __future__ import annotations

import math
import re

import polars as pl
import pytest

from src.config import REPO_ROOT
from src.data import assert_asof_respected
from src.evaluation import EPS, constant_log_loss, log_loss
from src.features import assert_logs_within_cutoff, build_features
from tests.conftest import NODATA, SLOW, make_synthetic_cohort

# 測試觀察期的上界。SPEC 紅線 7：不得使用 2017-04 之後的任何資料。
MAX_ALLOWED_DATE = 20170430


# ===========================================================================
# 紅線 1 · 任何 transaction_date > cutoff 的交易不得進入特徵
# ===========================================================================
# 「到期日之後的交易就是標籤」—— 這是本題的頭號洩漏。


@NODATA
def test_red_line_1_guard_catches_violation():
    """守門函式必須能擋下違規的表。

    這是「會失敗的測試」本體：故意造一張 b 用戶交易日晚於 cutoff 的表，
    確認 assert_asof_respected 真的會 raise。如果哪天有人把守門邏輯改壞
    （例如比較符號寫反），這個測試會紅。
    """
    bad = pl.DataFrame(
        {
            "msno": ["a", "b"],
            "cutoff": [20170228, 20170228],
            "last_tx": [20170227, 20170301],  # b 的最後一筆交易在 cutoff 之後
        }
    )
    with pytest.raises(AssertionError, match="紅線 1 違反"):
        assert_asof_respected(bad)


@NODATA
def test_red_line_1_guard_accepts_clean_table():
    """乾淨的表要能通過，否則守門太嚴會擋掉正常流程。

    邊界值 cutoff == last_tx 必須通過：用戶在到期日當天交易是合法的，
    那筆資料在預測時點確實看得到。
    """
    good = pl.DataFrame(
        {
            "msno": ["a", "b"],
            "cutoff": [20170228, 20170216],
            "last_tx": [20170201, 20170216],  # b 剛好等於 cutoff，邊界值
        }
    )
    assert_asof_respected(good)  # 不應 raise


@NODATA
def test_red_line_1_guard_rejects_missing_columns():
    """欄位缺失要明確報錯，不能默默視為通過。"""
    with pytest.raises(KeyError):
        assert_asof_respected(pl.DataFrame({"msno": ["a"]}))


@SLOW
@pytest.mark.parametrize("fixture_name", ["feb_cohort", "mar_cohort"])
def test_red_line_1_real_cohorts_are_clean(request, fixture_name: str):
    """實際產出的兩個 cohort 特徵表都必須通過守門。"""
    df = request.getfixturevalue(fixture_name)
    assert_asof_respected(df)


# ===========================================================================
# 紅線 3 · 禁用 members.csv，只能用 members_v3.csv
# ===========================================================================
# members.csv 的 expiration_date 是快照欄位，記錄「資料匯出當下」的到期日。
# 官方在 2017-11-13 發布 v3 就是為了移除它。


@NODATA
def test_red_line_3_no_code_reads_members_csv():
    """程式碼裡不得出現對 members.csv 的引用。

    只比對「被引號包起來的檔名」，不比對純文字 —— 註解裡寫「不能用
    members.csv」是說明，不是引用，不該讓測試失敗。
    """
    pattern = re.compile(r"""["']members\.csv["']""")
    offenders = []

    for folder in ("src", "scripts", "notebooks", "tests"):
        for path in (REPO_ROOT / folder).rglob("*.py"):
            if pattern.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, (
        f"這些檔案引用了 members.csv（紅線 3 禁用，只能用 members_v3.csv）：{offenders}"
    )


def test_red_line_3_v3_lacks_the_leaky_column(raw_files):
    """members_v3.csv 不得含 expiration_date。

    與 test_data_contract.py 的斷言 9 重複是刻意的：契約測試檢查的是
    「資料長得對不對」，這裡檢查的是「紅線有沒有守住」。兩份清單各自完整，
    刪掉任一支都不會讓紅線失去守門。
    """
    assert "expiration_date" not in raw_files("members_v3.csv").collect_schema().names()


# ===========================================================================
# 紅線 7 · 不得使用 2017-04（測試觀察期）之後的任何資料
# ===========================================================================


@SLOW
@pytest.mark.parametrize("fixture_name", ["feb_cohort", "mar_cohort"])
def test_red_line_7_no_future_dates(request, fixture_name: str):
    """特徵表裡所有日期欄位都不得晚於 2017-04-30。

    涵蓋 cutoff、first_tx、last_tx 與 registration_init_time。最後那個是
    容易漏掉的一個 —— 它來自 members_v3，不受 as-of 截斷保護。
    """
    df = request.getfixturevalue(fixture_name)
    date_cols = ["cutoff", "first_tx", "last_tx", "registration_init_time"]

    for col in date_cols:
        worst = df[col].max()
        if worst is None:  # 整欄皆 null（例如全部不在 members_v3）
            continue
        assert worst <= MAX_ALLOWED_DATE, f"{col} 最大值 {worst} 晚於 {MAX_ALLOWED_DATE}"


@SLOW
def test_red_line_7_cutoff_inside_cohort_window(feb_cohort, mar_cohort):
    """cutoff 必須落在該 cohort 宣告的到期區間內。

    這條擋的是「cohort 定義被改壞」—— 例如日期區間打錯一位數，
    模型還是會訓練成功，只是訓練在錯的族群上。
    """
    for df, (lo, hi) in ((feb_cohort, (20170201, 20170228)), (mar_cohort, (20170301, 20170331))):
        assert df["cutoff"].min() >= lo
        assert df["cutoff"].max() <= hi


# ===========================================================================
# 紅線 8 · 本地評估必須套用官方的 clip(1e-15, 1-1e-15)
# ===========================================================================


@NODATA
def test_red_line_8_clip_prevents_infinity():
    """極端預測必須產生有限值。

    沒有 clip 的話 ln(0) = -∞，整個評估變成 inf，一筆極端錯誤就摧毀所有
    資訊。官方的 clip 把單筆懲罰上限定在 -ln(1e-15) ≈ 34.54。
    """
    # 低端：y=1 而 p=0，p 被 clip 成 EPS。
    loss_low = log_loss([1], [0.0])
    assert math.isfinite(loss_low)
    assert loss_low == pytest.approx(-math.log(EPS), rel=1e-9)
    assert loss_low == pytest.approx(34.5387763949, rel=1e-9)

    # 高端：y=0 而 p=1，p 被 clip 成 1-EPS。
    #
    # ⚠️ 這一側**不等於** -ln(EPS)。1-1e-15 在 float64 裡無法精確表示：
    # 1 附近的 double 間距是 2^-53 ≈ 1.11e-16，所以 1-1e-15 會落在距離 1
    # 約 9.992e-16 的那個 double 上，而不是剛好 1e-15。結果是
    #     -ln(1 - (1-EPS)) ≈ 34.5396   （低端是 34.5388）
    #
    # 官方計分（sklearn 的 log_loss）用完全相同的 clip(eps, 1-eps)，
    # 因此有完全相同的不對稱。這裡照抄而不「修正」—— 紅線 8 要求的是
    # 與官方一致，不是數值上最漂亮。改用 log1p 讓兩側對稱，反而會讓
    # 本地分數與 LB 對不起來，正好違背這條紅線的目的。
    loss_high = log_loss([0], [1.0])
    assert math.isfinite(loss_high)
    assert loss_high == pytest.approx(-math.log(1.0 - (1.0 - EPS)), rel=1e-12)

    # 兩側幾乎相同但不完全相同。差異只在 p 落到極端值時才出現，
    # 對實際模型輸出（p 通常在 0.001~0.999）沒有影響。
    assert abs(loss_high - loss_low) < 1e-3


@NODATA
def test_red_line_8_matches_hand_computation():
    """與手算的公式對照，確認實作沒寫錯。

    公式：logloss = -(1/N) Σ [ y·ln(p) + (1-y)·ln(1-p) ]
    """
    y = [1, 0, 1, 0]
    p = [0.9, 0.1, 0.8, 0.3]
    expected = -sum(
        yi * math.log(pi) + (1 - yi) * math.log(1 - pi) for yi, pi in zip(y, p, strict=True)
    ) / len(y)
    assert log_loss(y, p) == pytest.approx(expected, rel=1e-12)


@NODATA
def test_red_line_8_rejects_bad_input():
    """長度不符或空輸入要明確報錯，不能靜靜回傳一個數字。"""
    with pytest.raises(ValueError):
        log_loss([1, 0], [0.5])
    with pytest.raises(ValueError):
        log_loss([], [])


@SLOW
def test_m1_baseline_threshold(feb_cohort, mar_cohort):
    """M1 驗收門檻必須是 0.30746（SPEC §3.3）。

    用 Feb cohort 的流失率對 Mar cohort 做常數預測。這個數字是 M1 的
    及格線 —— 打不贏它的模型沒有存在意義。把它鎖進測試，之後就不會
    有人記錯門檻。
    """
    feb_rate = feb_cohort["is_churn"].mean()
    baseline = constant_log_loss(feb_rate, mar_cohort["is_churn"])
    assert round(baseline, 5) == 0.30746


# ===========================================================================
# 尚未能實作的四條 —— 依賴還不存在的程式碼
# ===========================================================================
# 保留在這裡而不是刪掉，是為了讓清單保持完整：每次跑 pytest 都會列出
# 這四條 skip 與原因，提醒還欠什麼。用 `uv run pytest -ra` 可以看到。


@NODATA
def test_red_line_2_guard_catches_post_cutoff_logs():
    """守門函式必須擋下含 cutoff 之後日誌的特徵表。

    與紅線 1 同樣的測法：餵一張 u1 的最近日誌晚於自己 cutoff 的表，
    確認 assert_logs_within_cutoff 會 raise。

    `log_min_days_before` 是「最近一筆日誌距離 cutoff 幾天」，負值代表那筆
    日誌發生在到期日之後 —— 到期後的收聽行為是結果不是原因，讓它進特徵
    等於用未來預測過去。
    """
    bad = pl.DataFrame({"msno": ["u0", "u1"], "log_min_days_before": [0, -3]})
    with pytest.raises(AssertionError, match="紅線 2 違反"):
        assert_logs_within_cutoff(bad)


@NODATA
def test_red_line_2_guard_accepts_same_day_logs():
    """cutoff 當天的日誌必須通過（邊界值 0）。

    用戶在到期日當天聽歌是合法的，那筆資料在評分時點確實看得到。
    實測 Feb cohort 的 log_min_days_before 中位數就是 0 —— 多數人到期
    當天仍在使用，把 0 擋掉會誤殺一半以上的資料。
    """
    good = pl.DataFrame({"msno": ["u0", "u1"], "log_min_days_before": [0, 45]})
    assert_logs_within_cutoff(good)


@NODATA
def test_red_line_2_guard_rejects_missing_column():
    """缺欄位要明確報錯，不能默默視為通過。"""
    with pytest.raises(KeyError):
        assert_logs_within_cutoff(pl.DataFrame({"msno": ["u0"]}))


@SLOW
@pytest.mark.parametrize("cohort_name", ["feb", "mar"])
def test_red_line_2_real_log_features_are_clean(paths, cohort_name: str):
    """實際產出的兩個 cohort 收聽特徵表都必須通過守門。"""
    from src.features import build_log_features

    if not (paths.raw / "user_logs.csv").exists():
        pytest.skip("user_logs.csv 不存在，請執行 download.py --groups logs")
    assert_logs_within_cutoff(build_log_features(cohort_name, paths, verbose=False))


@pytest.mark.skip(reason="紅線 4：等 M3 —— 還沒有合併多 cohort 的程式碼")
def test_red_line_4_groupkfold_when_cohorts_merged():
    """合併多 cohort 做 KFold 時必須用 GroupKFold(groups=msno)。

    實測 90.81% 的用戶跨兩期出現（見 test_data_contract.test_cohort_overlap）。
    在合併資料上做隨機切分，會讓同一個人同時出現在訓練與驗證集。
    """


@NODATA
def test_red_line_5_feature_builder_is_stateless():
    """特徵建構不得依賴整批資料的統計量。

    **測法**：對完整資料建一次特徵，再對其中一個子集建一次，比對相同那幾列
    的值是否逐格相同。

    為什麼這樣測得出來：任何「從資料學來的」轉換 —— `fillna(df.mean())`、
    標準化、類別頻率編碼、target encoding —— 算出來的統計量都會隨輸入的
    列集合而變。子集的平均數不等於全集的平均數，於是同一位用戶在兩次呼叫
    中會得到不同的特徵值，這個測試就會紅。

    反過來說，只要這個測試是綠的，紅線 5 就不可能被違反 —— 因為根本沒有
    跨列的統計量存在，也就沒有東西可以從驗證集倒灌進訓練集。

    這比「小心翼翼地在每個 fold 內 fit」可靠得多。SPEC §5 註明這條
    「AI 產生的程式碼幾乎必犯」，而最好的防法是讓它無從犯起。

    ⚠️ M3 若引入 target encoding，本測試會失敗 —— 那是正確的行為。屆時
    編碼必須移進 fold 內的 pipeline，並由紅線 6 的測試接手守門。
    """
    full = make_synthetic_cohort()
    subset_idx = [0, 2, 5, 7]

    from_full = build_features(full).X[subset_idx]
    from_subset = build_features(full[subset_idx]).X

    assert from_full.columns == from_subset.columns
    assert from_full.equals(from_subset), (
        "同一位用戶在完整資料與子集上算出不同的特徵值 —— "
        "代表特徵建構用到了跨列的統計量，違反紅線 5。"
    )


@pytest.mark.skip(reason="紅線 6：等 M3 —— 還沒有 target encoding")
def test_red_line_6_target_encoding_is_oof():
    """target encoding 必須 out-of-fold。

    payment_method_id 是高基數類別（cohort 內實測 33 種），直接 target
    encode 會讓 CV 飆高、實測崩盤。
    """
