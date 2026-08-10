"""M3 · 讓 LightGBM / XGBoost / CatBoost 站在同一條起跑線上。

SPEC §7 的 M3 要求「三方比較」。要讓比較有意義，三個套件必須拿到**完全
相同的東西**：同一份特徵矩陣、同一個 Feb→Mar 時間外切分、同一塊 Feb 內部
early stopping 驗證集、同一個評估函式。只要有一項不同，分數差異就無法歸因
到「模型不一樣」。

## 為什麼需要這一層轉接

三個套件對「類別特徵」的要求互不相容，這是本模組存在的唯一理由：

| 套件 | 類別特徵怎麼給 | 缺失怎麼表示 |
|---|---|---|
| LightGBM | 欄位**索引** + 非負整數值 | **負值** = 缺失 |
| XGBoost  | pandas `category` dtype | `NaN`（類別欄不得出現負值） |
| CatBoost | `cat_features` + **字串或整數** | 不得為 `NaN`，缺失要自成一類 |

`src.features.build` 產出的類別欄一律是「非負整數，-1 代表缺失」（LightGBM
的約定）。所以：

  - XGBoost：-1 必須換成 null，否則 xgboost 會拒絕負的類別值
  - CatBoost：-1 保留，轉成字串 "-1" 自成一類

**這三種處理在語意上是等價的**（都表示「這個人不在 members_v3 裡」），差異
純粹是 API 慣例。若不轉接而是隨便挑一種餵給三家，比的就不是模型而是誰比較
會忍受錯誤格式的輸入。

## 為什麼統一介面回傳 `Fitted` 而不是各家的 booster

下游（比較表、null importance、MLflow、M5 原因碼）只需要四件事：預測、
停在第幾輪、特徵重要度、逐列 SHAP 歸因。把它們收斂成一個 dataclass，比較的
程式碼就完全不必知道底下是誰 —— 加第四個套件只要多寫一個 `fit_*` 函式。

⚠️ **SHAP 也走這一層，不把 booster 露出去。** M5 需要逐位用戶的歸因，最省事
的做法是讓 `Fitted` 帶著模型本體，解釋端自己算。但那等於要求解釋端把上面那張
轉接表**再寫一份**（CatBoost 要 `Pool` 加 `cat_features` 索引、XGBoost 要固定
字典的 `category` dtype、LightGBM 要無欄名的浮點矩陣）。兩份轉接遲早分歧，而
分歧的症狀是「歸因指到錯的欄位」—— 名單照樣產生、機率完全正確、原因碼張張
可讀，只是講的是別人的事，沒有任何一行程式會抱怨。

⚠️ **特徵重要度的數值不可跨套件比較。** 三家的 gain 定義不同（LightGBM 是
分裂增益總和、XGBoost 是 total_gain、CatBoost 是 PredictionValuesChange），
只有「同一個模型內部的排序」有意義。所以 `Fitted.importance` 一律附上
`gain_share`（佔比），比較時只看排名不看絕對值。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from src.features import FeatureSet

# `src.features.build.MISSING_CATEGORY` 的值。在這裡重新宣告是為了讓轉接
# 邏輯讀起來自足 —— 但用 import 綁住，避免兩邊各自改成不同的值。
from src.features.build import MISSING_CATEGORY


@dataclass(frozen=True)
class Fitted:
    """一個訓練完成的模型，以下游需要的最小介面呈現。"""

    name: str
    best_iteration: int
    predict: Callable[[pl.DataFrame], np.ndarray]
    importance: pl.DataFrame  # feature / gain / gain_share，已依 gain 遞減排序

    # 逐列 SHAP 歸因。輸出 `(列數, 特徵數 + 1)`，**最後一欄是 base value**
    # （模型對所有人的共同起點）。三家的 API 不同但形狀慣例一致：
    #
    #     LightGBM  booster.predict(pred_contrib=True)
    #     XGBoost   booster.predict(pred_contribs=True)
    #     CatBoost  model.get_feature_importance(type="ShapValues")
    #
    # 三者都是**精確 TreeSHAP**（多項式時間的樹上精確解），不是 KernelSHAP
    # 那種取樣近似 —— 所以「換一次執行結果會不一樣」這件事不存在。
    #
    # ⚠️ **單位是 log-odds，不是機率。** 恆等式是
    # `sigmoid(base + sum(shap)) == predict()`，相加不等於機率。把 +0.8 讀成
    # 「流失機率多 80%」是錯的；驗證與換算見 `src/explain/attribution.py`。
    shap_values: Callable[[pl.DataFrame], np.ndarray]

    def top(self, n: int = 12) -> pl.DataFrame:
        return self.importance.head(n)


def _importance_frame(features: list[str], gains: dict[str, float]) -> pl.DataFrame:
    """把 {特徵: gain} 整成統一格式，沒被用到的特徵補 0。

    補 0 這一步很重要：XGBoost 的 `get_score()` **只回傳有被分裂用過的特徵**，
    沒用到的直接不出現在字典裡。若不補回來，三家的重要度表會長度不同，而
    「這個特徵一次都沒被用到」正是 null importance 篩選最想知道的資訊。
    """
    total = sum(gains.values()) or 1.0
    return (
        pl.DataFrame(
            {
                "feature": features,
                "gain": [float(gains.get(f, 0.0)) for f in features],
            }
        )
        .with_columns((pl.col("gain") / total).alias("gain_share"))
        .sort("gain", descending=True)
    )


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------


def to_lgb_arrays(fs: FeatureSet) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """轉成 LightGBM 需要的 numpy 格式，並算出類別特徵的欄位索引。

    用索引而非欄名，是因為輸入是無欄名的 numpy 陣列。索引由 FeatureSet 的
    欄位順序推導，不寫死 —— 特徵順序改了也不會對錯欄位。
    """
    X = fs.X.to_numpy().astype(np.float64)
    y = fs.y.to_numpy().astype(np.int8)
    cat_idx = [fs.X.columns.index(c) for c in fs.categorical]
    return X, y, cat_idx


def fit_lightgbm(
    train: FeatureSet, es: FeatureSet, params: dict[str, Any], train_cfg: dict
) -> Fitted:
    """LightGBM。與 M1/M2 走的是同一條路徑，作為比較的參照點。"""
    import lightgbm as lgb

    X_tr, y_tr, cat_idx = to_lgb_arrays(train)
    X_es, y_es, _ = to_lgb_arrays(es)

    dtrain = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_idx, free_raw_data=False)
    dvalid = lgb.Dataset(X_es, y_es, categorical_feature=cat_idx, reference=dtrain)

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=train_cfg["num_boost_round"],
        valid_sets=[dvalid],
        valid_names=["inner"],
        callbacks=[lgb.early_stopping(train_cfg["early_stopping_rounds"], verbose=False)],
    )
    best = int(booster.best_iteration)
    names = train.X.columns

    def predict(X: pl.DataFrame) -> np.ndarray:
        return booster.predict(X.to_numpy().astype(np.float64), num_iteration=best)

    def shap_values(X: pl.DataFrame) -> np.ndarray:
        # `num_iteration=best` 要跟 predict 傳同一個值，否則歸因來自一棵比
        # 預測多幾輪的模型，加總恆等式會差一點點 —— 而「差一點點」正是最難
        # 察覺的那種錯。
        return booster.predict(
            X.to_numpy().astype(np.float64), num_iteration=best, pred_contrib=True
        )

    gains = dict(zip(names, booster.feature_importance("gain"), strict=True))
    return Fitted("LightGBM", best, predict, _importance_frame(names, gains), shap_values)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


def _to_xgb_pandas(fs_or_X: FeatureSet | pl.DataFrame, categorical: tuple[str, ...]):
    """轉成 XGBoost 的 `enable_categorical` 需要的 pandas DataFrame。

    兩件事一定要做對：

    1. **-1 換成 null。** XGBoost 要求類別值是非負整數，餵 -1 會直接 raise。
       換成 null 之後 XGBoost 走缺失分支，語意與 LightGBM 的負值一致。
    2. **category 的 dtype 要固定成同一組類別。** 訓練集與驗證集若各自
       `astype("category")`，同一個 `city=13` 可能被編到不同的內部碼 ——
       模型就會把兩批資料的類別對錯。所以下面用**明確的類別清單**建構
       dtype，清單由呼叫端統一提供。
    """
    X = fs_or_X.X if isinstance(fs_or_X, FeatureSet) else fs_or_X
    X = X.with_columns(
        [
            pl.when(pl.col(c) == MISSING_CATEGORY).then(None).otherwise(pl.col(c)).alias(c)
            for c in categorical
        ]
    )
    return X.to_pandas()


def xgb_category_levels(train: pl.DataFrame, *, categorical: tuple[str, ...]) -> dict[str, list]:
    """蒐集每個類別欄出現過的值，作為 pandas category 的固定字典。

    ⚠️ **只收一個 frame，而且必須是訓練資料。**

    簽名刻意是單一參數而非 `*frames`。可變參數版本讓
    `xgb_category_levels(feb.X, mar.X, ...)` 寫得出來 —— 而那一行看起來完全
    無害（「把兩邊的類別都收齊，才不會漏」），實際上就是下面說的
    transductive leakage。**能寫出來的錯，遲早有人會寫。** 收成單一參數之後，
    想混進驗證集會直接 TypeError，不必靠 code review 抓。

    這個字典是一個「擬合出來的前處理狀態」—— 它決定了哪些取值算是合法類別、
    以及每個取值對應到哪個內部碼。紅線 5 要求「所有 imputation / scaling /
    encoding 統計量必須在 fold 內計算」，類別字典就是這裡說的 encoding 狀態，
    因此它只能看訓練集。

    把驗證集也餵進來（`xgb_category_levels(feb.X, mar.X, ...)`）不會洩漏
    **標籤**，但會讓前處理器依驗證集的分布而定 —— 那是 transductive learning，
    在部署時做不到：正式上線時未來會出現哪些 `payment_method_id`，當下不可能
    知道。實測 Mar cohort 有一個 Feb 沒有的 `last_payment_method_id`（影響
    1 列），數值上微不足道，但作法上必須一致 —— 紅線的門檻訂在作法，不是
    訂在「這次影響大不大」（同樣的道理見 §7.4 的 target encoding 對照）。

    訓練時沒見過的類別在推論時會落到 pandas 的 NaN，XGBoost 走缺失分支 ——
    這與 LightGBM 對負值、CatBoost 對未知字串的處理一致，也正是部署時真正
    會發生的事。
    """
    levels: dict[str, list] = {}
    for c in categorical:
        seen = set(train[c].unique().to_list())
        seen.discard(MISSING_CATEGORY)
        seen.discard(None)
        levels[c] = sorted(seen)
    return levels


def fit_xgboost(
    train: FeatureSet,
    es: FeatureSet,
    params: dict[str, Any],
    train_cfg: dict,
    *,
    category_levels: dict[str, list],
) -> Fitted:
    """XGBoost（hist + enable_categorical）。

    `enable_categorical` 是 XGBoost 2.x 起的原生類別支援，作法與 LightGBM
    同源（在直方圖上找最佳類別切分），因此這組比較才是「同一種手法的不同
    實作」，而不是「原生類別 vs one-hot」這種被前處理決定勝負的比較。
    """
    import pandas as pd
    import xgboost as xgb

    def to_frame(fs: FeatureSet | pl.DataFrame):
        df = _to_xgb_pandas(fs, train.categorical)
        for c, levels in category_levels.items():
            df[c] = pd.Categorical(df[c], categories=levels)
        return df

    X_tr, X_es = to_frame(train), to_frame(es)
    dtrain = xgb.DMatrix(X_tr, train.y.to_numpy(), enable_categorical=True)
    dvalid = xgb.DMatrix(X_es, es.y.to_numpy(), enable_categorical=True)

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=train_cfg["num_boost_round"],
        evals=[(dvalid, "inner")],
        early_stopping_rounds=train_cfg["early_stopping_rounds"],
        verbose_eval=False,
    )
    # XGBoost 的 best_iteration 是 0-based，這裡統一成「輪數」（1-based），
    # 才能跟 LightGBM / CatBoost 的數字放在同一欄比較。
    best = int(booster.best_iteration) + 1

    def predict(X: pl.DataFrame) -> np.ndarray:
        d = xgb.DMatrix(to_frame(X), enable_categorical=True)
        return booster.predict(d, iteration_range=(0, best))

    def shap_values(X: pl.DataFrame) -> np.ndarray:
        d = xgb.DMatrix(to_frame(X), enable_categorical=True)
        return booster.predict(d, iteration_range=(0, best), pred_contribs=True)

    return Fitted(
        "XGBoost",
        best,
        predict,
        _importance_frame(train.X.columns, booster.get_score(importance_type="total_gain")),
        shap_values,
    )


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------


def fit_catboost(
    train: FeatureSet, es: FeatureSet, params: dict[str, Any], train_cfg: dict
) -> Fitted:
    """CatBoost。

    SPEC §8 選它的理由是「對高基數類別（`payment_method_id`，實測 33 種）
    有原生處理」。它的作法與另外兩家不同：**ordered target statistics** ——
    也就是內建的、依隨機排列逐列展開的 target encoding。

    這一點直接連到紅線 6（「target encoding 必須 out-of-fold」）：CatBoost
    的排序統計量在計算第 i 列的編碼時只用排在它前面的列，**結構上就不可能
    看到自己的標籤**。所以用 CatBoost 的原生類別處理不違反紅線 6；違反的是
    我們自己在 pipeline 外先算一份全表的類別平均數。這個對照見
    `src/features/encoding.py`。

    缺失值：CatBoost 的類別欄不接受 NaN，所以 -1 原樣保留並轉成字串
    `"-1"`，讓「不在 members_v3 裡」自成一類。這與 LightGBM 把負值視為
    缺失、XGBoost 走 NaN 分支在語意上一致。
    """
    from catboost import CatBoostClassifier, Pool

    cat_cols = list(train.categorical)

    def to_frame(fs: FeatureSet | pl.DataFrame):
        X = fs.X if isinstance(fs, FeatureSet) else fs
        return X.with_columns([pl.col(c).cast(pl.String) for c in cat_cols]).to_pandas()

    cat_idx = [train.X.columns.index(c) for c in cat_cols]
    train_pool = Pool(to_frame(train), train.y.to_numpy(), cat_features=cat_idx)
    es_pool = Pool(to_frame(es), es.y.to_numpy(), cat_features=cat_idx)

    model = CatBoostClassifier(
        **params,
        iterations=train_cfg["num_boost_round"],
        early_stopping_rounds=train_cfg["early_stopping_rounds"],
        verbose=False,
    )
    model.fit(train_pool, eval_set=es_pool, use_best_model=True)
    best = int(model.get_best_iteration()) + 1

    def predict(X: pl.DataFrame) -> np.ndarray:
        return model.predict_proba(Pool(to_frame(X), cat_features=cat_idx))[:, 1]

    def shap_values(X: pl.DataFrame) -> np.ndarray:
        # 走 predict 的同一條轉接（類別欄轉字串 + cat_features 索引）。
        #
        # 這是 M5 選 CatBoost 原生 TreeSHAP 而不裝 `shap` 套件的地方：同一個
        # 精確演算法、不多一個依賴，而且 Pool 的組法留在本模組 —— 解釋端不必
        # 知道「類別欄要先轉字串」這件事。
        #
        # `use_best_model=True` 已經讓模型只保留到 best_iteration，所以這裡
        # 不必也不能再指定輪數；歸因與預測必然來自同一棵模型。
        return model.get_feature_importance(
            Pool(to_frame(X), cat_features=cat_idx), type="ShapValues"
        )

    gains = dict(zip(train.X.columns, model.get_feature_importance(), strict=True))
    return Fitted("CatBoost", best, predict, _importance_frame(train.X.columns, gains), shap_values)
