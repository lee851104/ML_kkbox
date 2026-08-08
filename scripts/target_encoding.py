"""M3 · Target encoding 對照實驗（SPEC §5 紅線 6）。

紅線 6 說「target encoding 必須 out-of-fold，否則 CV 飆高、實測崩盤」。
這支腳本把那句話變成五個可比較的數字：

    A. 原生類別（LightGBM 直接處理 last_payment_method_id）        ← 對照組
    B. OOF TE · last_payment_method_id                             合規
    C. Naive TE · last_payment_method_id（用自己那批的標籤擬合）    ⚠️ 違規
    D. Naive TE · msno，切分**之後**才編碼                          ⚠️ 違規
    E. Naive TE · msno，切分**之前**就編碼                          ⚠️ 違規

C、D、E 是**故意寫錯的控制組**。SPEC §5 要求「每一條紅線都要有對應的會失敗
的測試」—— 單元測試證明守門函式擋得住合成資料，這支腳本則量出違規在真實
資料上的代價。

## 三個違規變體為什麼要分成三個

它們示範的是同一個錯誤在三種條件下的三種面貌，而這正是紅線 6 難懂的地方：

**C（低基數）**：`last_payment_method_id` 只有 33 種取值，每種平均兩萬多列。
把自己那一列的標籤算進去，只會讓類別平均值動 1/24000 —— 洩漏被大樣本稀釋
到量不出來。**違規在這裡幾乎不會被分數抓到。**

**D（高基數，切分後編碼）**：`msno` 每列一個唯一值，編碼幾乎就是自己的標籤。
但驗證集的 msno 從沒出現在訓練集的編碼表裡，只能拿到先驗 —— 模型在訓練集
學到的規則到驗證集完全失效，early stopping 立刻停手。**分數會當場崩掉。**

**E（高基數，切分前編碼）**：這才是紅線 6 描述的那個經典錯誤 —— 先對整個
訓練 cohort 做 target encoding，之後才切驗證集。驗證集的每一列也帶著自己的
標籤，於是**內部分數好得不像話，時間外分數崩盤**。C 與 E 的對比說明了為什麼
門檻要訂在「作法」而不是「分數有沒有變差」：分數看不出 C 的違規，但同一個
錯誤換到 E 就是災難，而兩者在程式碼裡長得幾乎一樣。

    uv run python scripts/target_encoding.py
    make encode
"""

from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from src.config import load_paths
from src.evaluation import log_loss
from src.features import FeatureSet
from src.features.encoding import fit_target_encoder, oof_target_encode
from src.models.candidates import fit_lightgbm
from src.models.train import load_cohort_features, load_model_config

TARGET_COL = "last_payment_method_id"

# 變體：(顯示名稱, 要編碼的欄位, 模式)
#   none          不編碼
#   oof           切分後，訓練集用 OOF 編碼
#   naive         切分後，訓練集用自己的標籤擬合再套回自己
#   naive_presplit 切分前，用整個 Feb cohort 擬合再套回整個 Feb
VARIANTS = (
    ("A 原生類別（對照）", None, "none"),
    (f"B OOF TE · {TARGET_COL}", TARGET_COL, "oof"),
    (f"C Naive TE · {TARGET_COL}", TARGET_COL, "naive"),
    ("D Naive TE · msno（切分後）", "msno", "naive"),
    ("E Naive TE · msno（切分前）", "msno", "naive_presplit"),
)


def with_encoded(fs: FeatureSet, col: str, encoded: pl.Series) -> FeatureSet:
    """把某欄換成它的編碼值，並把它從類別特徵清單裡移除。

    移除這一步不能忘：編碼後的欄位是連續的 [0, 1] 機率，若仍被宣告成類別
    特徵，LightGBM 會把每個不同的浮點數當成一個獨立類別 —— 那既沒有意義
    也會爆炸性地過擬合。
    """
    X = fs.X.drop(col) if col in fs.X.columns else fs.X
    return FeatureSet(
        X=X.with_columns(encoded.alias(f"{col}_te")),
        y=fs.y,
        msno=fs.msno,
        categorical=tuple(c for c in fs.categorical if c != col),
    )


def source_column(fs: FeatureSet, col: str) -> pl.Series:
    """取要編碼的原始欄。`msno` 不在特徵矩陣裡，要從 FeatureSet 另外拿。"""
    return fs.msno if col == "msno" else fs.X[col]


def make_split(feb: FeatureSet, train_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Feb 內部切分的索引。**五個變體共用同一批列**，否則分數不可比。"""
    return train_test_split(
        np.arange(feb.X.height),
        test_size=train_cfg["inner_valid_fraction"],
        random_state=train_cfg["inner_split_seed"],
        stratify=feb.y.to_numpy(),
    )


def prepare(
    feb: FeatureSet,
    mar: FeatureSet,
    tr_idx: np.ndarray,
    es_idx: np.ndarray,
    col: str | None,
    mode: str,
) -> tuple[FeatureSet, FeatureSet, FeatureSet]:
    """依模式產生 (訓練, early stopping, Mar) 三份特徵矩陣。"""
    if mode == "none":
        return feb.take(tr_idx), feb.take(es_idx), mar

    if mode == "naive_presplit":
        # 違規的關鍵在順序：**先**用整個 Feb（含之後會被切成驗證集的那些列）
        # 擬合編碼器，**再**切分。驗證集的每一列因此也帶著自己的標籤。
        src_feb = source_column(feb, col)
        encoder = fit_target_encoder(src_feb, feb.y)
        feb_v = with_encoded(feb, col, encoder.transform(src_feb))
        mar_v = with_encoded(mar, col, encoder.transform(source_column(mar, col)))
        return feb_v.take(tr_idx), feb_v.take(es_idx), mar_v

    # oof / naive：先切分，編碼器只看得到訓練集。
    train, es = feb.take(tr_idx), feb.take(es_idx)
    src_train = source_column(train, col)

    if mode == "oof":
        enc_train = oof_target_encode(src_train, train.y)
    elif mode == "naive":
        enc_train = fit_target_encoder(src_train, train.y).transform(src_train)
    else:  # pragma: no cover - VARIANTS 已窮舉
        raise ValueError(f"未知的模式 {mode}")

    # 驗證集與 Mar 一律用「整個訓練集」擬合的編碼器 —— 它們的標籤本來就不
    # 參與計算，不需要再切折。oof 與 naive 的唯一差別就是訓練集自己怎麼編碼。
    encoder = fit_target_encoder(src_train, train.y)
    return (
        with_encoded(train, col, enc_train),
        with_encoded(es, col, encoder.transform(source_column(es, col))),
        with_encoded(mar, col, encoder.transform(source_column(mar, col))),
    )


def main() -> int:
    try:
        paths = load_paths()
        cfg = load_model_config()
    except FileNotFoundError as e:
        sys.exit(str(e))

    params, train_cfg = dict(cfg["model"]), cfg["training"]
    feb, mar = load_cohort_features(paths, cfg)
    tr_idx, es_idx = make_split(feb, train_cfg)
    print(f"  Feb 內部切分：訓練 {len(tr_idx):,} · early stopping {len(es_idx):,}\n")

    rows = []
    for i, (name, col, mode) in enumerate(VARIANTS, 1):
        print(f"[{i}/{len(VARIANTS)}] {name}", flush=True)
        train_v, es_v, mar_v = prepare(feb, mar, tr_idx, es_idx, col, mode)

        t0 = time.perf_counter()
        fitted = fit_lightgbm(train_v, es_v, params, train_cfg)
        secs = time.perf_counter() - t0

        # **兩個分數都要報。** 只看內部分數會以為模型變強了，只看時間外分數
        # 會以為它只是沒用而已 —— 洩漏的特徵是「內部好、時間外差」的剪刀差。
        row = {
            "變體": name,
            "輪數": fitted.best_iteration,
            "Feb 內部": round(log_loss(es_v.y, fitted.predict(es_v.X)), 5),
            "Mar 時間外": round(log_loss(mar_v.y, fitted.predict(mar_v.X)), 5),
            "秒": round(secs),
        }
        rows.append(row)
        print(
            f"      Feb 內部 {row['Feb 內部']:.5f}　Mar 時間外 {row['Mar 時間外']:.5f}\n",
            flush=True,
        )

    table = pl.DataFrame(rows)
    ref_in, ref_out = table["Feb 內部"][0], table["Mar 時間外"][0]
    table = table.with_columns(
        (pl.col("Feb 內部") / ref_in - 1).round(4).alias("內部相對 A"),
        (pl.col("Mar 時間外") / ref_out - 1).round(4).alias("時間外相對 A"),
    )

    pl.Config.set_tbl_rows(10)
    pl.Config.set_tbl_width_chars(160)
    print("=" * 78)
    print("Target encoding 對照（SPEC §5 紅線 6）")
    print("=" * 78)
    print(table)
    print(
        "\n讀法：\n"
        "  B 與 A 幾乎相同 —— 對 33 種取值的類別，OOF target encoding 沒有帶來新資訊，\n"
        "     LightGBM 的原生類別處理已經夠好。\n"
        "  C 的違規**量不出來** —— 每個類別有兩萬多列，自己那一列的標籤只佔 1/24000。\n"
        "  D 與 E 是同一個違規換到高基數欄位（msno，每列一個唯一值）：\n"
        "     D 切分後才編碼 → 驗證集拿不到編碼，early stopping 立刻停手，兩邊都崩。\n"
        "     E 切分前就編碼 → 內部分數好得不像話、時間外崩盤，這就是紅線 6 說的剪刀差。\n"
        "\n結論：違規的代價由**欄位基數**與**編碼發生在切分的哪一側**決定，不是由\n"
        "「有沒有用 target encoding」決定。所以紅線 6 的門檻訂在作法上 —— 靠看分數\n"
        "來抓這個錯誤，在 C 那種情況下抓不到。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
