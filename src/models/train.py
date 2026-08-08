"""M1 · LightGBM baseline。

SPEC §4.2 的時間外驗證：

    訓練 Feb cohort（2017-02 到期，觀察期 2017-03）
    驗證 Mar cohort（2017-03 到期，觀察期 2017-04）

驗收門檻是 **Mar cohort 的 log loss < 0.30746**（SPEC §3.3），也就是「用 Feb
的流失率 6.3923% 對所有人做常數預測」的分數。打不贏它代表模型學到的東西
還不如「大家風險都一樣」這個假設。

## Early stopping 的驗證集為什麼不用 Mar

要停在第幾輪是一個**看著分數做的決定**。如果拿 Mar 來決定，回報的 Mar 分數
就已經被它自己影響過，會偏樂觀 —— 而那正是要跟 0.30746 比較、並外推到測試集
的數字。所以 early stopping 用 Feb 內部切出來的一小塊，Mar 全程不參與訓練，
保持乾淨的時間外身分。

在單一 cohort 內部做隨機分層切分是安全的：契約測試已驗證 `msno` 在一個
cohort 內不重複，且 as-of 截斷逐用戶計算，不存在時間洩漏。SPEC §4.2 也明文
允許「在 Feb cohort 內部另做 StratifiedKFold」。

執行入口在 `scripts/train.py`：

    uv run python scripts/train.py
    make train
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
import yaml
from sklearn.model_selection import train_test_split

from src.config import REPO_ROOT, Paths, load_paths
from src.data import FEB, MAR, build_cohort
from src.evaluation import constant_log_loss, log_loss, repeat_vs_new, segment_report
from src.features import FeatureSet, build_features

DEFAULT_CONFIG = REPO_ROOT / "configs" / "model_lgbm.yaml"

# SPEC §3.3 的 M1 驗收門檻。寫成常數是為了讓程式自己判斷有沒有過關，
# 而不是印個數字讓人自己比對 —— 人會看漏。
M1_THRESHOLD = 0.30746


@dataclass
class TrainResult:
    """一次訓練的完整結果，供報表與後續步驟使用。"""

    booster: lgb.Booster
    best_iteration: int
    logloss: float
    baseline_logloss: float
    segments: pl.DataFrame
    importance: pl.DataFrame
    valid_pred: np.ndarray

    @property
    def beats_baseline(self) -> bool:
        return self.logloss < self.baseline_logloss

    @property
    def improvement(self) -> float:
        """相對常數基準的改善比例。"""
        return 1 - self.logloss / self.baseline_logloss


def load_model_config(path: Path | None = None) -> dict[str, Any]:
    """讀 configs/model_lgbm.yaml。

    找不到就直接失敗，不套用預設值 —— 靜默的預設值會讓「我改了設定但沒生效」
    這種問題極難察覺。
    """
    path = path or DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"找不到模型設定檔 {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for section in ("model", "training"):
        if section not in cfg:
            raise KeyError(f"{path} 缺少 [{section}] 區段")
    return cfg


def _to_arrays(fs: FeatureSet) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """轉成 LightGBM 需要的 numpy 格式，並算出類別特徵的欄位索引。

    用索引而非欄名，是因為輸入是無欄名的 numpy 陣列。索引由 FeatureSet 的
    欄位順序推導，不寫死 —— 特徵順序改了也不會對錯欄位。
    """
    X = fs.X.to_numpy().astype(np.float64)
    y = fs.y.to_numpy().astype(np.int8)
    cat_idx = [fs.X.columns.index(c) for c in fs.categorical]
    return X, y, cat_idx


def train_baseline(
    paths: Paths | None = None,
    config: dict[str, Any] | None = None,
    *,
    verbose: bool = True,
) -> TrainResult:
    """在 Feb cohort 上訓練，在 Mar cohort 上評估。

    Returns:
        TrainResult，含 Mar cohort 的 log loss、分群報告與特徵重要度。
    """
    paths = paths or load_paths()
    cfg = config or load_model_config()
    params = dict(cfg["model"])
    train_cfg = cfg["training"]

    def log(msg: str = "") -> None:
        if verbose:
            print(msg, flush=True)

    # ---- 資料 ----
    log("載入 cohort...")
    feb_raw = build_cohort(FEB, paths, verbose=False)
    mar_raw = build_cohort(MAR, paths, verbose=False)
    feb = build_features(feb_raw)
    mar = build_features(mar_raw)
    log(f"  訓練 Feb {feb.X.height:,} 列 × {feb.X.width} 特徵，流失率 {feb.y.mean():.4%}")
    log(f"  驗證 Mar {mar.X.height:,} 列 × {mar.X.width} 特徵，流失率 {mar.y.mean():.4%}")

    if feb.X.columns != mar.X.columns:
        raise ValueError("兩個 cohort 的特徵欄位不一致，模型無法套用")

    X_feb, y_feb, cat_idx = _to_arrays(feb)
    X_mar, y_mar, _ = _to_arrays(mar)

    # ---- Feb 內部切一小塊給 early stopping（Mar 全程不參與訓練）----
    X_tr, X_es, y_tr, y_es = train_test_split(
        X_feb,
        y_feb,
        test_size=train_cfg["inner_valid_fraction"],
        random_state=train_cfg["inner_split_seed"],
        stratify=y_feb,
    )
    log(f"  Feb 內部切分：訓練 {len(y_tr):,} · early stopping {len(y_es):,}")

    # ---- 訓練 ----
    log("\n訓練中...")
    dtrain = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_idx, free_raw_data=False)
    des = lgb.Dataset(X_es, y_es, categorical_feature=cat_idx, reference=dtrain)

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=train_cfg["num_boost_round"],
        valid_sets=[des],
        valid_names=["feb_inner"],
        callbacks=[
            lgb.early_stopping(train_cfg["early_stopping_rounds"], verbose=verbose),
            lgb.log_evaluation(train_cfg["log_every_n"] if verbose else 0),
        ],
    )
    log(f"  最佳輪數 {booster.best_iteration}")

    # ---- 在 Mar cohort 上評估 ----
    pred = booster.predict(X_mar, num_iteration=booster.best_iteration)
    ll = log_loss(y_mar, pred)
    baseline = constant_log_loss(float(feb.y.mean()), y_mar)

    # ---- SPEC §4.5 要求的分群回報 ----
    scored = pl.DataFrame(
        {
            "msno": mar.msno,
            "is_churn": mar.y,
            "p_churn": pred,
            "segment": repeat_vs_new(mar.msno, feb.msno),
        }
    )
    segments = segment_report(scored, segment_col="segment")

    importance = (
        pl.DataFrame(
            {
                "feature": feb.X.columns,
                "gain": booster.feature_importance("gain"),
                "split": booster.feature_importance("split"),
            }
        )
        .with_columns((pl.col("gain") / pl.col("gain").sum()).alias("gain_share"))
        .sort("gain", descending=True)
    )

    return TrainResult(
        booster=booster,
        best_iteration=booster.best_iteration,
        logloss=ll,
        baseline_logloss=baseline,
        segments=segments,
        importance=importance,
        valid_pred=pred,
    )


def _print_report(r: TrainResult) -> None:
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(140)

    print("\n" + "=" * 70)
    print("M1 · LightGBM baseline 結果")
    print("=" * 70)
    print(f"最佳輪數              {r.best_iteration}")
    print(f"常數基準 log loss     {r.baseline_logloss:.5f}   ← SPEC §3.3 門檻")
    print(f"模型 log loss         {r.logloss:.5f}")
    print(f"改善                  {r.improvement:.2%}")
    print(f"驗收                  {'✅ 通過' if r.beats_baseline else '❌ 未達門檻'}")

    print("\n--- 分群回報（SPEC §4.5 要求）---")
    print(r.segments)

    print("\n--- 特徵重要度 Top 12（依 gain）---")
    print(r.importance.head(12))

    zero = r.importance.filter(pl.col("gain") == 0)
    if zero.height:
        print(f"\n完全沒被用到的特徵 {zero.height} 個：{zero['feature'].to_list()}")


def main() -> int:
    try:
        result = train_baseline()
    except FileNotFoundError as e:
        sys.exit(f"{e}\n請先執行 uv run python scripts/download.py")

    _print_report(result)

    if not result.beats_baseline:
        print(f"\n❌ Mar cohort log loss {result.logloss:.5f} 未打敗基準 {M1_THRESHOLD}")
        return 1
    return 0
