"""`/docs` 的具名範例必須是自洽的。

範例是 Demo 的全部 —— 訪客不會讀 README，只會下拉選一個然後按 Execute。所以
一組算錯的範例比沒有範例更糟：它會安靜地展示一個矛盾的輸入，而畫面上看不出來。

這裡守三件事：

  1. **衍生欄位與基礎計數一致。** completion / active_ratio / secs_per_active_day
     與三個 trend 都是算出來的，公式在 src/features/logs.py。那邊改了而
     examples.py 沒跟上，這裡要失敗。
  2. **窗口是巢狀的。** 7 天的計數不可能大於 14 天的，以此類推。手寫數字時
     很容易破壞這個關係。
  3. **cutoff 落在模型見過的區間。** 落在外面不會被服務拒絕，但那是外插，
     機率不可信 —— Demo 不該示範一個不可信的用法。
"""

from __future__ import annotations

import pytest

from src.features.logs import LOG_WINDOWS
from src.serving.examples import OPENAPI_EXAMPLES, WINDOWS

ALL = list(OPENAPI_EXAMPLES.items())
WITH_LOGS = [(k, v) for k, v in ALL if v["value"].get("logs")]


def test_windows_match_the_feature_layer():
    """examples.py 重列了一份 WINDOWS，不是 import 的。對不上就在這裡爆。"""
    assert tuple(WINDOWS) == tuple(LOG_WINDOWS)


@pytest.mark.parametrize("key,ex", ALL)
def test_every_example_carries_a_summary_and_description(key, ex):
    assert ex["summary"].strip()
    assert len(ex["description"]) > 80, f"{key} 的說明太短，Demo 上等於沒說"


@pytest.mark.parametrize("key,ex", WITH_LOGS)
def test_derived_log_fields_match_their_formulas(key, ex):
    logs = ex["value"]["logs"]
    for w in WINDOWS:
        plays = logs[f"log{w}_plays"]
        active = logs[f"log{w}_active_days"]
        secs = logs[f"log{w}_secs"]
        completed = logs[f"log{w}_completed"]

        expected_completion = round(completed / plays, 4) if plays > 0 else None
        assert logs[f"log{w}_completion"] == expected_completion, f"{key} log{w}_completion"

        assert logs[f"log{w}_active_ratio"] == round(active / w, 4), f"{key} log{w}_active_ratio"

        expected_spad = round(secs / active, 1) if active > 0 else None
        assert logs[f"log{w}_secs_per_active_day"] == expected_spad, f"{key} log{w}_secs_per_day"


@pytest.mark.parametrize("key,ex", WITH_LOGS)
def test_trends_match_their_formula(key, ex):
    logs = ex["value"]["logs"]

    def expected(num: str, num_days: int, den: str, den_days: int):
        per_day_den = logs[den] / den_days
        if not per_day_den > 0:
            return None
        return round((logs[num] / num_days) / per_day_den, 4)

    assert logs["log_trend_7_30"] == expected("log7_secs", 7, "log30_secs", 30), key
    assert logs["log_trend_30_90"] == expected("log30_secs", 30, "log90_secs", 90), key
    assert logs["log_trend_active_7_30"] == expected(
        "log7_active_days", 7, "log30_active_days", 30
    ), key


@pytest.mark.parametrize("key,ex", WITH_LOGS)
def test_windows_are_nested(key, ex):
    """7 ⊆ 14 ⊆ 30 ⊆ 90。短窗口的計數不可能超過長窗口。"""
    logs = ex["value"]["logs"]
    for metric in ("active_days", "secs", "plays", "completed", "unq"):
        values = [logs[f"log{w}_{metric}"] for w in WINDOWS]
        for short, long, sw, lw in zip(values, values[1:], WINDOWS, WINDOWS[1:], strict=False):
            assert short <= long, f"{key}: log{sw}_{metric}={short} > log{lw}_{metric}={long}"


@pytest.mark.parametrize("key,ex", WITH_LOGS)
def test_active_days_cannot_exceed_the_window(key, ex):
    logs = ex["value"]["logs"]
    for w in WINDOWS:
        assert logs[f"log{w}_active_days"] <= w, f"{key}: log{w}_active_days 超過 {w} 天"


@pytest.mark.parametrize("key,ex", ALL)
def test_transaction_dates_are_ordered(key, ex):
    f = ex["value"]["features"]
    assert f["first_tx"] <= f["last_tx"], f"{key}: first_tx 晚於 last_tx"
    assert f["last_tx"] <= f["cutoff"], f"{key}: last_tx 晚於 cutoff —— 那是紅線 1"
    if f.get("registration_init_time"):
        assert f["registration_init_time"] <= f["first_tx"], f"{key}: 註冊日晚於首次交易"


@pytest.mark.parametrize("key,ex", ALL)
def test_cutoff_is_inside_the_window_the_model_was_trained_on(key, ex):
    """artifact 的 cohort.cutoff_window。載不到 artifact 就跳過 —— CI 沒有它。"""
    try:
        from src.serving.app import load_serving_config
        from src.serving.artifact import load_artifact

        # 與 app 的 lifespan 同一條路徑：名稱來自 configs/serving.yaml，
        # 環境變數 MODEL_ARTIFACT 若有設會優先。不帶參數呼叫會 ValueError。
        art = load_artifact(name=load_serving_config()["artifact"])
    except Exception as exc:  # noqa: BLE001 - 環境問題，不是測試失敗
        pytest.skip(f"載不到 artifact：{type(exc).__name__}: {exc}")

    lo, hi = art.summary()["cohort"]["cutoff_window"]
    cutoff = ex["value"]["features"]["cutoff"]
    assert lo <= cutoff <= hi, f"{key}: cutoff {cutoff} 落在訓練區間 [{lo}, {hi}] 之外"


def test_the_demo_tells_a_coherent_story():
    """已取消的那位必須明顯高於忠誠用戶。

    不斷言絕對數值 —— 模型重新匯出後會變。斷言的是**排序**：如果這兩個
    反過來，Demo 展示的就是一個壞掉的模型，而那比沒有 Demo 糟糕得多。
    """
    try:
        from fastapi.testclient import TestClient

        from src.serving.app import app
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"起不了服務：{type(exc).__name__}")

    with TestClient(app) as client:
        scores = {}
        for key, ex in ALL:
            r = client.post("/predict", json=ex["value"])
            assert r.status_code == 200, f"{key} 回 {r.status_code}：{r.text[:200]}"
            scores[key] = r.json()["p_churn"]

    assert scores["cancelled"] > scores["loyal"] * 10, (
        f"已取消 {scores['cancelled']:.4f} 沒有明顯高於忠誠用戶 {scores['loyal']:.4f}"
    )
    assert scores["cancelled"] > scores["autorenew_off"] > scores["loyal"], (
        f"風險排序不符預期：{scores}"
    )
