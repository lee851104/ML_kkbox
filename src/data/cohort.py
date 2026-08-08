"""建立 as-of cohort 特徵表 —— 本專案的防洩漏核心。

SPEC §4.3 的定義：

    cutoff(user) = membership_expire_date(user)
    所有進入特徵的 transaction_date 必須 <= cutoff

為什麼標籤會洩漏：`is_churn` 的定義是「到期後 30 天內沒有新的有效訂閱交易」。
而 transactions_v2.csv 涵蓋到 2017-03-31。對一個 2017-02 到期的用戶來說，
**他 3 月那筆交易就是答案本身**。讓它進特徵，CV 分數會漂亮到不真實，上線後全崩。

為什麼這段程式碼要獨立成模組，而不是留在 notebook 裡：

  1. **同一段邏輯要跑兩次。** M1 用 Feb cohort 訓練、Mar cohort 驗證。
     複製貼上再改日期，是紅線 1 最容易破功的地方 —— 改了一處忘了另一處，
     而且不會有任何錯誤訊息。
  2. **它可以被測試。** notebook 裡的程式碼寫不了 pytest。
  3. **截斷條件只寫在一個地方。** 要改就一起改。

本模組還內建了一個永遠會跑的守門檢查（見 build_cohort 末段）：算完之後
驗證沒有任何用戶的最後一筆交易晚於自己的 cutoff。這不是測試，是產線程式碼
的一部分 —— 洩漏一旦發生，寧可整支爆掉也不要靜靜地產出錯的表。
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.config import Paths, load_paths


@dataclass(frozen=True)
class CohortSpec:
    """一個到期月份的 cohort 定義。

    Attributes:
        name:         用於快取檔名。
        label_file:   官方標籤檔（raw/ 底下）。
        expire_start: 到期日區間下界（含），格式 %Y%m%d 的整數。
        expire_end:   到期日區間上界（含）。
        observation:  標籤的觀察期，僅供人閱讀與報表標示。
    """

    name: str
    label_file: str
    expire_start: int
    expire_end: int
    observation: str


# SPEC §4.2 的時間外驗證切分：
#     訓練 Feb cohort（觀察期 2017-03）→ 驗證 Mar cohort（觀察期 2017-04）
# 兩者的時間結構同構，所以驗證分數可以外推到測試表現。
FEB = CohortSpec("feb", "train.csv", 20170201, 20170228, "2017-03")
MAR = CohortSpec("mar", "train_v2.csv", 20170301, 20170331, "2017-04")

COHORTS: dict[str, CohortSpec] = {c.name: c for c in (FEB, MAR)}

# 快取的 schema。欄位對不上就重算 —— 比要求使用者手動刪快取好，
# 因為「忘記刪快取所以看到舊結果」是很難察覺的錯誤。
EXPECTED_COLUMNS = frozenset(
    {
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
)


def scan_transactions(paths: Paths) -> pl.LazyFrame:
    """兩個交易檔的聯集，lazy。

    **兩個都要載入。** SPEC §5.1 的避雷清單：transactions_v2.csv 有 74.76%
    是 2017-03 的交易，它是增量更新檔不是完整歷史。只用 v2 等於完全沒有
    歷史特徵，而模型還是會訓練成功 —— 只是分數很差且找不出原因。
    """
    return pl.concat(
        [
            pl.scan_csv(paths.raw / "transactions.csv"),
            pl.scan_csv(paths.raw / "transactions_v2.csv"),
        ]
    )


def _last(col: str) -> pl.Expr:
    """cutoff 之前最後一筆交易的某個欄位。

    用 sort_by("transaction_date").last() 明確指定排序依據，而不是先 sort
    整張表再靠 group_by 保留順序 —— polars 不保證 group_by 內的列順序，
    那樣寫在小資料上會對、在大資料上偶爾錯，是最難查的一種 bug。
    """
    return pl.col(col).sort_by("transaction_date").last().alias(f"last_{col}")


def assert_asof_respected(df: pl.DataFrame) -> None:
    """紅線 1 守門：沒有任何用戶的最後一筆交易晚於自己的 cutoff。

    抽成獨立函式而不是寫在 build_cohort 裡面，是為了讓它可以被單獨測試。
    SPEC §5 要求每條紅線都要有「會失敗的測試」—— 測試必須能餵給它一張
    確實違規的表，確認它真的會 raise。只驗證「正常資料會通過」證明不了
    守門有效，因為一個永遠回傳 None 的空函式也會通過。

    Raises:
        AssertionError: 存在 last_tx > cutoff 的列。
    """
    missing = {"last_tx", "cutoff"} - set(df.columns)
    if missing:
        raise KeyError(f"缺少檢查所需的欄位：{sorted(missing)}")

    violations = df.filter(pl.col("last_tx") > pl.col("cutoff"))
    if violations.height:
        worst = violations.select((pl.col("last_tx") - pl.col("cutoff")).max().alias("d")).item()
        raise AssertionError(
            f"紅線 1 違反：{violations.height:,} 位用戶的特徵含 cutoff 之後的交易"
            f"（最嚴重的超出 cutoff 約 {worst} 天）。as-of 截斷失效，本表不可使用。"
        )


def build_cohort(
    spec: CohortSpec | str = FEB,
    paths: Paths | None = None,
    *,
    force: bool = False,
    verbose: bool = True,
) -> pl.DataFrame:
    """算出某個 cohort 每人一列的 as-of 特徵表。

    Args:
        spec:    CohortSpec，或 "feb" / "mar"。
        paths:   路徑設定，預設讀 configs/paths.yaml。
        force:   True 則忽略快取重算。
        verbose: 是否印進度。

    Returns:
        每位用戶一列。欄位見 EXPECTED_COLUMNS。

    Raises:
        AssertionError: 若產出的特徵含 cutoff 之後的交易（紅線 1 破功）。
    """
    if isinstance(spec, str):
        spec = COHORTS[spec]
    paths = (paths or load_paths()).ensure()

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    cache = paths.interim / f"{spec.name}_cohort_asof.parquet"
    if cache.exists() and not force:
        cached = pl.read_parquet(cache)
        if EXPECTED_COLUMNS <= set(cached.columns):
            # ⚠️ **快取命中也要跑守門。**
            #
            # 快取是一個 parquet 檔，不是一個保證。它可能是舊版程式產出的、
            # 可能被手動覆蓋、可能來自另一台機器 —— 而唯一會發現的時機就是
            # 現在。只檢查欄位存在等於只驗證「形狀對」，形狀對而內容洩漏的
            # 表會一路通到模型裡，且分數會**變好**，因此不會有人起疑。
            #
            # 這一步的成本是掃兩欄比大小，相對於重算整個 cohort 微不足道。
            assert_asof_respected(cached)
            log(f"讀取快取 {cache.name}（{cached.height:,} 列）")
            return cached
        log(f"快取 {cache.name} 的欄位與目前的 schema 不符，重算")

    log(f"建立 {spec.name} cohort（到期 {spec.expire_start}~{spec.expire_end}）...")

    labels = pl.read_csv(paths.raw / spec.label_file)
    tx = scan_transactions(paths)

    # ---- 步驟 1：每位用戶的 cutoff ----
    # cutoff = 落在本 cohort 到期區間內的 membership_expire_date。
    # 取 max 是因為同一個月內可能有多筆交易（例如月中改方案），最後那個
    # 才是真正的到期日。
    #
    # 第一個 filter 順帶擋掉了 SPEC §2.1 的兩個哨兵值：19700101（Unix epoch，
    # 等同 null）和 20361015（2036 年）都不在區間內，不會被選為 cutoff。
    #
    # ⚠️ **第二個 filter 是 as-of 截斷的一部分，不是資料清理。**
    #
    # 少了它，cutoff 本身就可能來自未來：實測有 158,766 筆交易的
    # transaction_date 晚於自己宣告的 membership_expire_date（95.93% 是取消
    # 紀錄，屬於補登慣例）。這種列在到期日當下**還不存在**，卻可以憑著它
    # 攜帶的到期日替用戶製造出一個 cutoff。
    #
    # 這是紅線 1 的守門抓不到的洩漏形式：`assert_asof_respected()` 檢查的是
    # 「last_tx <= cutoff」，而這裡出問題的是 **cutoff 這個基準點自己**。
    # 基準點錯了，所有以它為準的檢查都會通過，卻全部建立在未來資訊上。
    #
    # 判準：一筆交易在它自己宣告的到期日當下可觀測，等價於
    # `transaction_date <= membership_expire_date`。這個條件不循環（不依賴
    # 尚未算出的 cutoff），而且一旦成立，`transaction_date <= cutoff` 也自動
    # 成立。
    #
    # 代價：找不到任何「當下可觀測」交易的用戶會離開 cohort（實測 Feb 19 人、
    # Mar 1 人），另有 Feb 156 / Mar 148 人的 cutoff 值改變。這是正確的行為 ——
    # 部署時同樣算不出他們的 cutoff，硬留著等於假裝當時知道未來。
    cutoffs = (
        tx.filter(pl.col("membership_expire_date").is_between(spec.expire_start, spec.expire_end))
        .filter(pl.col("transaction_date") <= pl.col("membership_expire_date"))
        .group_by("msno")
        .agg(pl.col("membership_expire_date").max().alias("cutoff"))
        .collect(engine="streaming")
    )

    cohort = labels.join(cutoffs, on="msno", how="inner")
    log(f"  標籤 {labels.height:,} 人 → 對得上 cutoff {cohort.height:,} 人")

    # ---- 步驟 2：as-of 聚合 ----
    # 這一行 filter 就是紅線 1 本身。
    asof = (
        tx.join(cohort.lazy(), on="msno", how="inner")
        .filter(pl.col("transaction_date") <= pl.col("cutoff"))
        .group_by("msno")
        .agg(
            pl.col("is_churn").first(),
            pl.col("cutoff").first(),
            pl.len().alias("n_tx"),
            pl.col("transaction_date").min().alias("first_tx"),
            pl.col("transaction_date").max().alias("last_tx"),
            pl.col("is_cancel").sum().alias("n_cancel_hist"),
            pl.col("actual_amount_paid").mean().alias("mean_paid"),
            _last("is_cancel"),
            _last("is_auto_renew"),
            _last("actual_amount_paid"),
            _last("plan_list_price"),
            _last("payment_plan_days"),
            _last("payment_method_id"),
        )
    )

    # ---- 步驟 3：接上用戶屬性 ----
    # 用 left join 而不是 inner join：實測 11.66% 的 cohort 用戶不在
    # members_v3 裡，而「查不到」本身有訊號（那群人流失率 5.02%，低於
    # 整體 6.39%）。用 inner join 會把這 11.6 萬人整批丟掉。
    #
    # 只能用 members_v3.csv，不能用 members.csv（紅線 3）—— 後者含
    # expiration_date 快照欄位，官方發布 v3 就是為了移除那個洩漏欄位。
    out = (
        asof.join(pl.scan_csv(paths.raw / "members_v3.csv"), on="msno", how="left")
        .with_columns(pl.col("city").is_not_null().alias("in_members"))
        .collect(engine="streaming")
    )

    # ---- 步驟 4：紅線 1 守門檢查 ----
    # 永遠會跑，不是只在測試裡。洩漏一旦發生，寧可整支爆掉也不要靜靜產出錯的表。
    assert_asof_respected(out)

    out.write_parquet(cache)
    log(f"  完成 {out.height:,} 列 × {out.width} 欄，已快取 → {cache.name}")
    return out
