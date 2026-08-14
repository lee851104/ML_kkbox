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

    最後一筆已取消     流失率 85.70%   佔用戶 2.4%
    自動續訂關閉       流失率 32.25%   佔用戶 11.2%
    其餘               流失率  0.80%   佔用戶 86.4%
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
        # ⚠️ 自動續訂必須是 1，不是 0。
        #
        # 這兩欄不是獨立的：**沒開自動續訂的人根本不需要取消**，時間到就自然
        # 結束。實測 Feb cohort 99.2 萬人，`已取消 × 自動續訂關` 這個組合
        # **一筆都不存在**（見 reports/figures/03_cancel_x_autorenew_heatmap.png）。
        #
        # 這裡原本寫 0，於是這個「最高風險」的範例是一個訓練資料裡零支撐的
        # 組合 —— 模型照樣回一個看起來很合理的 0.8253，沒有任何地方會報錯。
        # 改成 1（24,303 人、流失率 85.70% 的真實分群）之後是 0.9652，而且
        # 取消那句原因碼的強度從 +2.39 升到 +4.11：模型真正學過的是「主動
        # 取消一個原本會自動扣款的訂閱」，不是這個拼出來的組合。
        "last_is_auto_renew": 1,
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
        "summary": "① 高風險：已經按過取消",
        "description": (
            "**這類人只佔全部用戶的 2.4%，但每 7 個就有 6 個會離開** —— "
            "而且他們一群人就佔了全部流失量的三分之一。\n\n"
            "兩個壞消息同時出現：主動按過取消、最近一週完全沒聽歌"
            "（雖然三個月前還很活躍）。五個範例裡機率最高的一個。\n\n"
            "⚠️ 注意他的自動續訂是**開著**的，這不是筆誤：沒開自動續訂的人"
            "根本不需要取消，時間到就自然結束。所以「已取消 × 自動續訂關」"
            "這個組合在 99.2 萬人裡一筆都不存在 —— 取消之所以是最強的訊號，"
            "正因為它是唯一需要用戶**主動做一件事**的狀態。\n\n"
            "⚠️ 另一個陷阱：取消這個動作常常就發生在到期當天。"
            "而真正能上線的模型必須**提前七天**判斷 —— 那時候還看不到這個訊號。"
            "提前七天的代價是準確度掉約 18%，這個數字專案裡有量。"
        ),
        "value": _CANCELLED,
    },
    "autorenew_off": {
        "summary": "② 中風險：還沒退訂，但已經不用了",
        "description": (
            "**這類人佔全部用戶的 11.2%，大約每 3 個有 1 個會離開。**"
            "因為人數多，他們佔了全部流失量的 56% —— 比 ① 那群還多。\n\n"
            "帳面上一切正常，沒按過取消。但自動續訂關著，而且最近一週只打開過一天，"
            "「整首聽完」的比例掉到剩三分之一。\n\n"
            "跟 ① 對照就看得出來：**光看一個開關不夠**，模型讀的是一整組行為。"
        ),
        "value": _AUTORENEW_OFF,
    },
    "loyal": {
        "summary": "③ 低風險：訂了兩年，每天都在聽",
        "description": (
            "**這類人佔全部用戶的 85.8%，1000 個裡只有 6 個會離開。**\n\n"
            "訂閱兩年、26 次扣款、自動續訂開著，而且不論看最近 7 天、14 天、30 天"
            "還是 90 天，聽歌的量都差不多 —— 沒有變冷的跡象。\n\n"
            "把這個跟 ① 輪流點一次，是最快看懂模型在做什麼的方法：機率會差上百倍。\n\n"
            "⚠️ 不過這群人不能因為風險低就忽略。他們人數實在太多，"
            "那 0.6% 加起來仍然是全部流失量的 8%。而且**用開關規則抓不出他們** —— "
            "這正是需要模型、而不是寫幾個 if 判斷的理由。"
        ),
        "value": _LOYAL,
    },
    "new_subscriber": {
        "summary": "④ 新客：整群很危險，不代表每個人都危險",
        "description": (
            "第一次訂閱剛好到期的人，只扣過一次款。"
            "**這類人平均每 5 個就有 2 個會離開，是老訂戶的近 7 倍。**\n\n"
            "但這一位跑出來的機率會**遠低於**那個平均值，因為他的自動續訂是開著的。\n\n"
            "**這個落差是故意留著的。** 模型看的是這個人的行為，"
            "不是「他被分在新客這一組」。如果照組別發優惠（「所有新客都發」），"
            "那就不需要機器學習了，寫個 if 判斷就好。模型的價值在於回答"
            "**同樣是新客，誰該發、誰不用發**。\n\n"
            "（他確實有風險訊號：買了之後只用五天，最近兩週完全沒打開。"
            "這些會出現在下方的原因排序裡，只是還壓不過自動續訂那一項。）"
        ),
        "value": _NEW_SUBSCRIBER,
    },
    "no_listening": {
        "summary": "⑤ 沒有收聽資料：看服務怎麼說實話",
        "description": (
            "**這一筆刻意不提供收聽紀錄**，訓練資料裡有 18% 的用戶就是這樣。\n\n"
            "重點不在機率，在下方跳出來的**警告訊息**：服務會告訴你"
            "「沒給收聽資料不等於中性，這等於在主張這個人 90 天沒聽過歌」，"
            "並附上那個 18% 讓你自己判斷這個假設合不合理。\n\n"
            "**一個會告訴你它假設了什麼的模型，比一個安靜給答案的模型有用。**\n\n"
            "這一筆的年齡欄位也是 0 —— 那是原始資料裡的無效值，服務不會替你猜一個。"
        ),
        "value": _NO_LISTENING,
    },
}
