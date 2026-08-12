"""`/predict` 的具名範例 —— Swagger UI 上那個下拉選單。

## 為什麼需要

`/docs` 是這個服務唯一的介面（見 app.py 開頭）。沒有範例時，Swagger 的
「Try it out」只能依欄位型別產生骨架，整數全填 `0` —— 而 `cutoff` 是 `%Y%m%d`
的日期，`0` 不是合法值。訪客按下 Execute 得到的是錯誤，不是 Demo。

要自己填對，得先知道：日期格式、T−7 模型的 cutoff 是「到期日減 7 天」、
日期得落在模型訓練的區間、`last_payment_method_id` 填幾合理、什麼組合算高風險。
前四項文件有寫，最後一項要讀完整份 README。沒有人會做這件事。

## 資料是合成的

⚠️ **這裡沒有任何一列真實資料。** 依競賽規則，KKBox 資料與其衍生特徵不得散布
（見 MODEL_CARD.md 的授權聲明、deploy/serving.yaml 關閉 msno 介面的理由）。

每個原型的數值是**依 README 已公開的統計分佈手造**的：

    最後一筆已取消     流失率 75.07%   佔用戶 2.9%
    自動續訂關閉       流失率 32.59%   佔用戶 11.3%
    其餘               流失率  0.63%   佔用戶 85.8%
    首次到期的新客     流失率 39.84%   佔 Mar cohort 9.19%
    完全沒有收聽紀錄   訓練資料裡 18.0% 的用戶

## 衍生欄位由程式算，不手寫

38 個收聽欄位裡有 12 個是比率（completion、active_ratio、secs_per_active_day）
與 3 個是趨勢。手寫的話，「完播數 118 / 播放數 140」與「完播率 0.62」很容易
對不上 —— 那會讓模型看到自相矛盾的輸入，而 Demo 上看不出來。

所以這裡只手寫**基礎計數**，比率照 `src/features/logs.py` 的 `_derived()` 與
`_ratio()` 同一組公式算出來。公式改了而這裡沒跟上，
`tests/test_examples.py` 會失敗。
"""

from __future__ import annotations

from typing import Any

# src/features/logs.py 的 LOG_WINDOWS。在這裡重列一次而不是 import，是為了讓
# 這個模組不依賴特徵層 —— 對不上時由測試抓，不是由 import 順序決定。
WINDOWS = (7, 14, 30, 90)


def _log_block(
    per_window: dict[int, dict[str, float]],
    *,
    min_days_before: float,
    max_days_before: float,
) -> dict[str, float | None]:
    """由基礎計數展開成 38 欄。

    公式與 `src/features/logs.py` 一致：
        completion          = completed / plays          （plays 為 0 時 null）
        active_ratio        = active_days / w
        secs_per_active_day = secs / active_days         （active 為 0 時 null）
        trend_a_b           = (a / a_days) / (b / b_days)（分母為 0 時 null）
    """
    out: dict[str, float | None] = {}
    for w in WINDOWS:
        base = per_window[w]
        active, secs = base["active_days"], base["secs"]
        plays, completed, unq = base["plays"], base["completed"], base["unq"]
        out[f"log{w}_active_days"] = active
        out[f"log{w}_secs"] = secs
        out[f"log{w}_plays"] = plays
        out[f"log{w}_completed"] = completed
        out[f"log{w}_unq"] = unq
        out[f"log{w}_completion"] = round(completed / plays, 4) if plays > 0 else None
        out[f"log{w}_active_ratio"] = round(active / w, 4)
        out[f"log{w}_secs_per_active_day"] = round(secs / active, 1) if active > 0 else None

    def trend(num: str, num_days: int, den: str, den_days: int) -> float | None:
        per_day_den = out[den] / den_days  # type: ignore[operator]
        if not per_day_den > 0:
            return None
        return round((out[num] / num_days) / per_day_den, 4)  # type: ignore[operator]

    out["log_trend_7_30"] = trend("log7_secs", 7, "log30_secs", 30)
    out["log_trend_30_90"] = trend("log30_secs", 30, "log90_secs", 90)
    out["log_trend_active_7_30"] = trend("log7_active_days", 7, "log30_active_days", 30)
    out["log_min_days_before"] = min_days_before
    out["log_max_days_before"] = max_days_before
    out["log_has_logs"] = 1.0
    return out


# ---------------------------------------------------------------------------
# 五個原型
# ---------------------------------------------------------------------------
#
# cutoff 一律落在 20170125–20170221 —— artifact 的 cohort.cutoff_window。
# 落在區間外不會被拒絕，但那是模型沒見過的時點，機率不可信。

_CANCELLED = {
    "features": {
        "cutoff": 20170214,
        "n_tx": 9,
        "first_tx": 20160312,
        "last_tx": 20170208,
        "n_cancel_hist": 2,
        "mean_paid": 149.0,
        "last_is_cancel": 1,
        "last_is_auto_renew": 0,
        "last_actual_amount_paid": 0.0,
        "last_plan_list_price": 149.0,
        "last_payment_plan_days": 30,
        "last_payment_method_id": 40,
        "city": 5,
        "bd": 31,
        "gender": "male",
        "registered_via": 9,
        "registration_init_time": 20160310,
    },
    "logs": _log_block(
        {
            # 近一週完全停止，但 90 天前是活躍的 —— 典型的「先冷掉再退訂」
            7: {"active_days": 0, "secs": 0.0, "plays": 0, "completed": 0, "unq": 0},
            14: {"active_days": 1, "secs": 900.0, "plays": 6, "completed": 3, "unq": 5},
            30: {"active_days": 6, "secs": 9800.0, "plays": 61, "completed": 38, "unq": 44},
            90: {"active_days": 41, "secs": 118000.0, "plays": 720, "completed": 480, "unq": 390},
        },
        min_days_before=12,
        max_days_before=88,
    ),
}

_AUTORENEW_OFF = {
    "features": {
        # 到期日 20170214，30 天方案 → 上一次續訂約 20170115（cutoff 前 23 天）。
        # 這三個日期要對得起來，否則 days_since_last_tx 會講出一個與方案週期
        # 矛盾的故事，而那正是模型最看重的特徵之一。
        "cutoff": 20170207,
        "n_tx": 14,
        "first_tx": 20151120,
        "last_tx": 20170115,
        "n_cancel_hist": 0,
        "mean_paid": 149.0,
        "last_is_cancel": 0,
        "last_is_auto_renew": 0,
        "last_actual_amount_paid": 149.0,
        "last_plan_list_price": 149.0,
        "last_payment_plan_days": 30,
        "last_payment_method_id": 38,
        "city": 13,
        "bd": 27,
        "gender": "female",
        "registered_via": 7,
        "registration_init_time": 20151118,
    },
    "logs": _log_block(
        {
            # 漸行漸遠：90 天前很活躍，近兩週剩零星，近一週只剩一天且完播率掉下來。
            # 這是「還沒退訂但已經不用了」的樣子。
            7: {"active_days": 1, "secs": 620.0, "plays": 9, "completed": 3, "unq": 8},
            14: {"active_days": 3, "secs": 2900.0, "plays": 26, "completed": 12, "unq": 22},
            30: {"active_days": 9, "secs": 14000.0, "plays": 104, "completed": 58, "unq": 81},
            90: {"active_days": 58, "secs": 141000.0, "plays": 910, "completed": 604, "unq": 520},
        },
        min_days_before=5,
        max_days_before=89,
    ),
}

_LOYAL = {
    "features": {
        # 到期日 20170208，30 天方案 → 上一次自動續訂 20170109。
        "cutoff": 20170201,
        "n_tx": 26,
        "first_tx": 20150118,
        "last_tx": 20170109,
        "n_cancel_hist": 0,
        "mean_paid": 149.0,
        "last_is_cancel": 0,
        "last_is_auto_renew": 1,
        "last_actual_amount_paid": 149.0,
        "last_plan_list_price": 149.0,
        "last_payment_plan_days": 30,
        "last_payment_method_id": 41,
        "city": 1,
        "bd": 35,
        "gender": "male",
        "registered_via": 9,
        "registration_init_time": 20150115,
    },
    "logs": _log_block(
        {
            # 幾乎天天聽，完播率高且四個窗口一致 —— 沒有衰退訊號
            7: {"active_days": 7, "secs": 29000.0, "plays": 140, "completed": 118, "unq": 96},
            14: {"active_days": 14, "secs": 57000.0, "plays": 275, "completed": 232, "unq": 180},
            30: {"active_days": 29, "secs": 122000.0, "plays": 590, "completed": 498, "unq": 340},
            90: {"active_days": 86, "secs": 361000.0, "plays": 1750, "completed": 1470, "unq": 820},
        },
        min_days_before=0,
        max_days_before=89,
    ),
}

_NEW_SUBSCRIBER = {
    "features": {
        # 首購後第一次到期。n_tx = 1 且 first_tx == last_tx 是這個族群的指紋。
        # 首購 20170108，30 天方案 → 到期 20170207，T−7 的 cutoff 是 20170131。
        "cutoff": 20170131,
        "n_tx": 1,
        "first_tx": 20170108,
        "last_tx": 20170108,
        "n_cancel_hist": 0,
        "mean_paid": 149.0,
        "last_is_cancel": 0,
        "last_is_auto_renew": 1,
        "last_actual_amount_paid": 149.0,
        "last_plan_list_price": 149.0,
        "last_payment_plan_days": 30,
        "last_payment_method_id": 41,
        "city": 22,
        "bd": 24,
        "gender": None,  # 缺失率 65.43%，而且「沒填」本身有訊號
        "registered_via": 4,
        "registration_init_time": 20170102,
    },
    "logs": _log_block(
        {
            # 衝動購買型：買了之後試用幾天就沒再打開。只有 23 天歷史，
            # 所以 30 天與 90 天窗口的數字相同 —— 這本身就是「新客」的指紋。
            7: {"active_days": 0, "secs": 0.0, "plays": 0, "completed": 0, "unq": 0},
            14: {"active_days": 0, "secs": 0.0, "plays": 0, "completed": 0, "unq": 0},
            30: {"active_days": 5, "secs": 5400.0, "plays": 38, "completed": 14, "unq": 35},
            90: {"active_days": 5, "secs": 5400.0, "plays": 38, "completed": 14, "unq": 35},
        },
        min_days_before=17,
        max_days_before=22,
    ),
}

_NO_LISTENING = {
    # 刻意不給 logs。訓練資料裡 18.0% 的用戶就是這個樣子，而回應會帶一句
    # 警告說明「省略不是中性預設，是一個主張」—— 那句警告本身值得被看到。
    "features": {
        # 到期日 20170223，30 天方案 → 上一次續訂 20170124。
        "cutoff": 20170216,
        "n_tx": 5,
        "first_tx": 20160820,
        "last_tx": 20170124,
        "n_cancel_hist": 1,
        "mean_paid": 149.0,
        "last_is_cancel": 0,
        "last_is_auto_renew": 0,
        "last_actual_amount_paid": 149.0,
        "last_plan_list_price": 149.0,
        "last_payment_plan_days": 30,
        "last_payment_method_id": 38,
        "city": 15,
        "bd": 0,  # 無效值。README：填 0 的人流失率 4.76%，比填了有效年齡的低
        "gender": None,
        "registered_via": 3,
        "registration_init_time": 20160818,
    },
}


# ⚠️ 說明文字裡引用的百分比一律是**分群的平均流失率**（來自 README 的 M0 分析），
# 不是「這個範例會輸出的機率」。兩者本來就不同 —— 模型看的是這 61 個特徵的組合，
# 不是這個人屬於哪一群。④ 就是刻意留著的反例：群體平均 39.84%，個體卻很低。
#
# 也刻意不把實際輸出的數字寫進說明。模型重新匯出後那些數字會變，而寫死的說明
# 不會跟著變 —— 一份與程式不同步的文件比沒有文件更糟。

OPENAPI_EXAMPLES: dict[str, dict[str, Any]] = {
    "cancelled": {
        "summary": "① 高風險：最後一筆交易已取消",
        "description": (
            "README 的 A 群 —— 佔用戶 2.9%，**該分群平均流失率 75.07%**，"
            "涵蓋全體流失量的 34.2%。\n\n"
            "訊號是疊加的：取消動作 ＋ 自動續訂關閉 ＋ 近 7 天收聽完全歸零"
            "（90 天前還很活躍）。這是五個範例裡機率最高的一個。\n\n"
            "⚠️ 取消常發生在到期日當天，而 T−7 模型的 cutoff 是到期日前 7 天 —— "
            "這個訊號在真正能上線的版本裡會大幅減弱。README「M6 的第一塊」量化了"
            "這個代價（log loss 從 0.15173 退到 0.17921，+18.11%）。"
        ),
        "value": _CANCELLED,
    },
    "autorenew_off": {
        "summary": "② 中風險：自動續訂關閉，且正在流失興趣",
        "description": (
            "README 的 B 群 —— 佔用戶 11.3%，**該分群平均流失率 32.59%**，"
            "涵蓋全體流失量的 57.4%（比 A 群還多，因為人數多得多）。\n\n"
            "沒有取消動作，帳面上還是正常訂戶。但自動續訂關著，而且近 7 天只剩一天"
            "有打開、完播率掉到三分之一 —— 「還沒退訂但已經不用了」。\n\n"
            "與 ① 對照可以看出：**單一旗標不等於高風險**，模型讀的是行為的組合。"
        ),
        "value": _AUTORENEW_OFF,
    },
    "loyal": {
        "summary": "③ 低風險：兩年自動續訂，收聽穩定",
        "description": (
            "README 的 C 群 —— 佔用戶 85.8%，**該分群平均流失率 0.63%**。\n\n"
            "2015 年起 26 筆交易、自動續訂開啟、四個時間窗口的收聽強度與完播率一致，"
            "看不到任何衰退。\n\n"
            "把這個與 ① 並排跑一次，是理解這個模型最快的方式 —— 機率會差好幾個量級。\n\n"
            "⚠️ 但 C 群不是可以忽略的：他們人數佔 85.8%，那 0.63% 仍然貢獻了 8.4% 的"
            "流失量。規則（看旗標）找不出這批人，這才是模型存在的理由。"
        ),
        "value": _LOYAL,
    },
    "new_subscriber": {
        "summary": "④ 首次到期的新客 —— 群體平均與個體預測不是同一件事",
        "description": (
            "只有一筆交易、`first_tx == last_tx`。這個族群佔 Mar cohort 的 9.19%，"
            "**該分群平均流失率 39.84%** —— 是重複用戶（5.87%）的 6.8 倍。\n\n"
            "但這一位的輸出會**遠低於** 39.84%。原因是他的自動續訂開著，而模型讀的是"
            "行為特徵，不是「他屬於新客這一群」。\n\n"
            "**這個落差是刻意留著的。** 拿群體基礎率當個體預測是最常見的分析誤用，"
            "而挽回名單如果照分群發（「對所有新客發優惠」），就退化成不需要機器學習的"
            "規則。模型的價值在**同一分群內部的排序能力** —— 這一位不該被投放，"
            "同一群裡自動續訂關著又沒在聽的人才該。\n\n"
            "（他確實有風險訊號：買了之後只用了 5 天，近兩週完全沒打開。看原因碼會"
            "出現在排序裡，只是不足以蓋過自動續訂。）"
        ),
        "value": _NEW_SUBSCRIBER,
    },
    "no_listening": {
        "summary": "⑤ 沒有收聽資料 —— 看服務怎麼說實話",
        "description": (
            "刻意不提供 `logs`。訓練資料裡 18.0% 的用戶就是這個樣子。\n\n"
            "重點不在機率，在回應的 `warnings`：服務會明講「省略收聽特徵**不是中性預設**，"
            "而是在主張這個人近 90 天沒聽過歌」，並附上那個 18.0% 讓你判斷這個主張合不合理。\n\n"
            "**一個會告訴你它假設了什麼的模型，比一個安靜給答案的模型有用。** "
            "這一筆的 `bd`（年齡）也是 0 —— 那是資料裡的無效值，服務不會替你猜。"
        ),
        "value": _NO_LISTENING,
    },
}
