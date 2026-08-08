"""把 as-of cohort 表轉成模型可用的特徵矩陣。

## 兩個核心設計決定

**一、本模組完全無狀態（stateless）。**

它不從資料計算任何統計量 —— 沒有平均數、沒有標準差、沒有類別頻率、沒有
target encoding。所有轉換都只用「這一列自己的值」和寫死的常數。

這是對紅線 5（「所有 imputation / scaling / encoding 統計量必須在 fold 內
計算」）**最強的回應：讓它無從違反**。一個不計算統計量的函式，不可能把
驗證集的資訊倒灌進訓練集。SPEC §5 註明「AI 產生的程式碼幾乎必犯」這條，
而最可靠的防法不是小心翼翼地在 fold 內 fit，是根本不需要 fit。

之所以做得到，是因為 LightGBM 本身就不需要 scaling、原生處理缺失值、
也原生處理類別特徵。M3 若引入 target encoding，那時才需要真正的 fold 內
pipeline，屆時紅線 6 的測試會派上用場。

**二、輸出不含任何原始日期。**

`cutoff`、`first_tx`、`last_tx`、`registration_init_time` 全部轉成「距離
cutoff 幾天」。理由是部署現實：訓練集的日期落在 2017-02，測試集在 2017-04，
兩者沒有交集。把原始日期餵進去，模型會學到「2017 年 2 月」這種在測試集
不存在的切點，訓練分數漂亮而測試崩盤。

⚠️ 日期相減必須先轉成真正的日期型別。`20170301 - 20170228 = 73`，不是 1 天
—— YYYYMMDD 是十進位編碼，不是天數。這個錯誤不會報錯，只會產生垃圾特徵。
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

# LightGBM 原生支援的類別特徵。
#
# 依 LightGBM 的約定，類別特徵必須是非負整數，且**所有負值一律視為缺失**。
# 因此下方把 null 填成 -1 —— 這不是「用 -1 這個值代替缺失」，而是明確地
# 告訴 LightGBM 這裡是缺失。實測 11.66% 的 cohort 用戶不在 members_v3 中，
# 他們的 city / registered_via / gender 全部落在這一類。
CATEGORICAL: tuple[str, ...] = (
    "city",
    "registered_via",
    "last_payment_method_id",
    "gender_code",
)

MISSING_CATEGORY = -1

# gender 的編碼是**寫死的對照表**，不是從資料學來的，所以無狀態。
# 缺失獨立成一類：實測缺失者流失率 4.86%，而男 8.84% / 女 8.64% 兩者
# 幾乎沒有差異 —— 「有沒有填」比「填什麼」有訊號得多。
GENDER_CODES = {"male": 0, "female": 1}

# bd（年齡）的合理範圍。官方明示此欄含 -7168 ~ 2016 的離群值，
# cohort 內只有 39.18% 落在此區間（SPEC §5.1 要求做對照實驗，M3 處理）。
BD_MIN, BD_MAX = 10, 100


@dataclass(frozen=True)
class FeatureSet:
    """一個 cohort 的特徵矩陣與標籤。

    X 保持 polars DataFrame 而非 numpy，是為了讓特徵在訓練前仍可檢視
    （欄名、dtype、分布）。轉成 numpy 在訓練模組的邊界才做。
    """

    X: pl.DataFrame
    y: pl.Series
    msno: pl.Series
    categorical: tuple[str, ...] = CATEGORICAL

    @property
    def names(self) -> list[str]:
        return self.X.columns

    def __repr__(self) -> str:  # pragma: no cover - 只影響顯示
        return f"FeatureSet({self.X.height:,} 列 × {self.X.width} 特徵, 流失率 {self.y.mean():.4%})"


def _as_date(col: str) -> pl.Expr:
    """把 %Y%m%d 的整數欄位轉成日期型別。

    strict=False：無法解析的值回 null 而不是整支炸掉。SPEC §2.1 提到的哨兵值
    19700101 其實是合法日期（Unix epoch），會正常解析 —— 它在 cutoff 的計算
    階段就已被 cohort 的日期區間過濾掉，不會進到這裡。
    """
    return pl.col(col).cast(pl.Int64).cast(pl.String).str.to_date("%Y%m%d", strict=False)


def _days_before_cutoff(col: str, alias: str) -> pl.Expr:
    """該日期距離 cutoff 幾天。負值（日期晚於 cutoff）一律轉成 null。

    為什麼要擋負值：`registration_init_time` 來自 members_v3 快照，**不受
    as-of 截斷保護**。實測 Feb cohort 有 6 人、Mar cohort 有 2 人的註冊日
    晚於自己的 cutoff（例如一位有 21 筆交易的用戶「註冊」於 2017-03）。
    數量微不足道，但那是未來資訊，餵負數進模型等於開一個小洞給紅線 7。
    轉成 null 讓 LightGBM 當缺失處理，既不洩漏也不製造假訊號。
    """
    days = (_as_date("cutoff") - _as_date(col)).dt.total_days()
    return pl.when(days >= 0).then(days).otherwise(None).alias(alias)


def build_features(df: pl.DataFrame, logs: pl.DataFrame | None = None) -> FeatureSet:
    """把 `src.data.build_cohort()` 的輸出轉成特徵矩陣。

    Args:
        df:   build_cohort 產生的 as-of 表，欄位見 `src.data.cohort.EXPECTED_COLUMNS`。
        logs: `src.features.build_log_features()` 的輸出（M2）。給了就以 left join
              併入。**必須是 left join** —— 實測 18.4% 的 cohort 用戶在 90 天窗口
              內沒有任何收聽紀錄，inner join 會把他們整批丟掉。

    Returns:
        FeatureSet。X 不含 `msno` 與任何原始日期欄位。

    Raises:
        KeyError: 輸入缺少必要欄位。
    """
    required = {
        "msno",
        "is_churn",
        "cutoff",
        "n_tx",
        "first_tx",
        "last_tx",
        "n_cancel_hist",
        "mean_paid",
        "last_is_cancel",
        "last_is_auto_renew",
        "last_actual_amount_paid",
        "last_plan_list_price",
        "last_payment_plan_days",
        "last_payment_method_id",
        "city",
        "bd",
        "gender",
        "registered_via",
        "registration_init_time",
        "in_members",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"輸入缺少欄位：{sorted(missing)}")

    paid = pl.col("last_actual_amount_paid")
    price = pl.col("last_plan_list_price")
    plan_days = pl.col("last_payment_plan_days")

    X = df.select(
        # --- 時間特徵：一律相對於 cutoff，不留原始日期 ---
        _days_before_cutoff("first_tx", "tenure_days"),
        _days_before_cutoff("last_tx", "days_since_last_tx"),
        _days_before_cutoff("registration_init_time", "days_since_registration"),
        # --- 交易史 ---
        # 實測交易史長度與流失率單調負相關：1-2 筆 26.52% → 25+ 筆 3.36%。
        pl.col("n_tx").cast(pl.Float64),
        pl.col("n_cancel_hist").cast(pl.Float64),
        (pl.col("n_cancel_hist") / pl.col("n_tx")).alias("cancel_rate"),
        pl.col("mean_paid"),
        # --- cutoff 之前最後一筆交易 ---
        # last_is_cancel 是實測最強的旗標（流失率 75.07% vs 4.33%），但它
        # 幾乎等於標籤：取消常發生在到期日當天，而 cutoff 就是到期日。
        # M6 的 cutoff = 到期日 − 7 天 版本會失去這個訊號，屆時分數必然下降。
        pl.col("last_is_cancel").cast(pl.Float64),
        pl.col("last_is_auto_renew").cast(pl.Float64),
        paid.cast(pl.Float64).alias("last_paid"),
        price.cast(pl.Float64).alias("last_price"),
        plan_days.cast(pl.Float64).alias("last_plan_days"),
        (price - paid).cast(pl.Float64).alias("discount_amount"),
        # 方案天數為 0 的交易實測有 12,861 筆，除法要擋掉否則產生 inf。
        pl.when(plan_days > 0).then(paid / plan_days).otherwise(None).alias("price_per_day"),
        # 這兩個旗標把「實付 0 元」拆成語意不同的兩件事（SPEC §2.1）：
        # 免費方案本來就收 0 元，與定價非 0 卻沒收到錢完全不同。
        (price == 0).cast(pl.Float64).alias("is_free_plan"),
        ((paid == 0) & (price > 0)).cast(pl.Float64).alias("zero_collected"),
        # --- 用戶屬性 ---
        # in_members 本身有訊號：查不到的那 11.66% 流失率 5.02%，低於整體 6.39%。
        pl.col("in_members").cast(pl.Float64),
        # bd 只有 39.18% 落在合理範圍。離群值不截斷也不補值，直接設 null 讓
        # LightGBM 走缺失分支；另外保留「原本是不是有效值」當獨立特徵。
        pl.when(pl.col("bd").is_between(BD_MIN, BD_MAX))
        .then(pl.col("bd"))
        .otherwise(None)
        .cast(pl.Float64)
        .alias("bd_clean"),
        pl.col("bd").is_between(BD_MIN, BD_MAX).fill_null(False).cast(pl.Float64).alias("bd_valid"),
        # --- 類別特徵（LightGBM 原生處理，負值代表缺失）---
        pl.col("city").fill_null(MISSING_CATEGORY).cast(pl.Int32),
        pl.col("registered_via").fill_null(MISSING_CATEGORY).cast(pl.Int32),
        pl.col("last_payment_method_id").cast(pl.Int32),
        pl.col("gender")
        .replace_strict(GENDER_CODES, default=MISSING_CATEGORY, return_dtype=pl.Int32)
        .alias("gender_code"),
    )

    if logs is not None:
        X = _attach_logs(df["msno"], X, logs)

    return FeatureSet(X=X, y=df["is_churn"], msno=df["msno"])


def _attach_logs(msno: pl.Series, X: pl.DataFrame, logs: pl.DataFrame) -> pl.DataFrame:
    """把收聽特徵 left join 到交易特徵上。

    沒有收聽紀錄的用戶：`log_has_logs` 填 0，其餘 log 欄位保持 null 讓
    LightGBM 走缺失分支。**不補 0** —— 「沒有紀錄」和「聽了 0 秒」是不同的
    兩件事，補 0 會把前者偽裝成後者。

    實測「完全沒有紀錄」這件事本身**幾乎沒有訊號**（Feb 6.12% vs 6.45%，
    Mar 方向甚至相反），推測是自動續訂的休眠訂戶：不聽但錢照扣，所以不流失。
    保留 `log_has_logs` 讓模型自己決定要不要用。

    這一步仍然是無狀態的：每位用戶的 log 特徵只由他自己的日誌決定，與批次
    裡有哪些人無關，所以紅線 5 的無狀態測試依然成立。
    """
    if "msno" not in logs.columns:
        raise KeyError("收聽特徵表缺少 msno 欄位，無法 join")

    joined = (
        pl.DataFrame({"msno": msno})
        .join(logs, on="msno", how="left")
        .with_columns(pl.col("log_has_logs").fill_null(0.0))
        .drop("msno")
    )
    if joined.height != X.height:
        raise ValueError(f"join 後列數改變（{X.height} → {joined.height}），收聽特徵有重複的 msno")

    return pl.concat([X, joined], how="horizontal")
