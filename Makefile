# KKBox 訂閱流失預測 —— 常用指令
#
# 每個目標都是單一 `uv run ...`，不使用 shell 內建指令（rm、cp、find 等），
# 這樣同一份 Makefile 在 Linux（CI）與 Windows 上行為一致。
#
# ⚠️ Windows 沒有內建 make。若未安裝，直接執行各目標底下的 uv 指令即可，
#    效果完全相同 —— 本檔案只是那些指令的集中索引。安裝方式見 README。

.DEFAULT_GOAL := help
.PHONY: help setup data data-all lint format test test-fast test-ci eda features ablation train compare select encode tune mlflow clean eval serve

help:
	@echo "可用目標："
	@echo ""
	@echo "  setup      建立環境（取得 Python 3.12、建 .venv、editable 安裝本專案）"
	@echo "  data       下載 M0/M1 需要的資料（約 1.02 GB）"
	@echo "  data-all   下載全部競賽資料（8.95 GB，解壓後約 34 GB）"
	@echo ""
	@echo "  lint       ruff 檢查 + 格式檢查（不改檔案）"
	@echo "  format     ruff 自動排版（會改檔案）"
	@echo ""
	@echo "  test       全部測試（需要資料，約 30 秒）"
	@echo "  test-fast  跳過需掃大檔的測試"
	@echo "  test-ci    只跑不需資料的純邏輯測試，要求零 skip"
	@echo ""
	@echo "  eda        產生 EDA 圖表到 reports/figures/"
	@echo "  features   M2 收聽行為聚合（首次約 30 秒，之後讀快取）"
	@echo "  ablation   收聽特徵的分組消融實驗"
	@echo "  train      訓練並評估（含 5-fold 標準差與 MLflow 追蹤）"
	@echo ""
	@echo "  compare    M3 三方比較：LightGBM / XGBoost / CatBoost（約 6 分鐘）"
	@echo "  select     M3 null importance 特徵篩選 + 篩選前後對照（約 20 分鐘）"
	@echo "  encode     M3 target encoding 對照（紅線 6，約 1 分鐘）"
	@echo "  tune       M3 LightGBM 隨機搜尋 31 組（約 15 分鐘）"
	@echo ""
	@echo "  mlflow     開啟 MLflow UI 檢視實驗紀錄"
	@echo "  clean      清除 __pycache__ / .pytest_cache / .ruff_cache"
	@echo ""
	@echo "  eval / serve   尚未實作，見各目標訊息"

# --- 環境與資料 ------------------------------------------------------------

setup:
	uv sync

data:
	uv run python scripts/download.py

data-all:
	uv run python scripts/download.py --groups all

# --- 程式碼品質 ------------------------------------------------------------

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .

# --- 測試 ------------------------------------------------------------------

test:
	uv run pytest

test-fast:
	uv run pytest -m "not slow"

# CI 專用。GitHub Actions 上沒有原始資料（資料不進 Git），43 條測試會有 36 條
# skip —— 而一個全部 skip 的套件也會顯示綠燈。所以 CI 單獨跑標了 nodata 的
# 純邏輯測試，並要求它們全過。零 skip 的檢查在 ci.yml 裡執行。
test-ci:
	uv run pytest -m nodata

# --- 分析與訓練 ------------------------------------------------------------

eda:
	uv run python notebooks/eda_01_overview.py

# M2 收聽行為聚合。第一次要掃 31.9 GB 原始日誌（約 30 秒），之後讀快取。
features:
	uv run python scripts/features.py

# 收聽特徵的分組消融。七次訓練，約 2 分鐘。
ablation:
	uv run python scripts/ablation.py

# M1 baseline。訓練 Feb cohort、在 Mar cohort 評估，未達 SPEC §3.3 的
# 0.30746 門檻會回傳非零離開碼。
train:
	uv run python scripts/train.py

# --- M3 ---------------------------------------------------------------------

# 三方比較。三家拿到同一份特徵、同一個切分、同一塊 early stopping 驗證集。
# CatBoost 佔掉大部分時間（對稱樹 + ordered target statistics 較慢）。
compare:
	uv run python scripts/compare.py

# Null importance 特徵篩選。真實 1 次 + 打亂標籤 20 次 + 各門檻重訓一次。
select:
	uv run python scripts/select_features.py

# Target encoding 對照（含三個刻意違規的控制組，證明紅線 6 不是空話）。
encode:
	uv run python scripts/target_encoding.py

# 超參數隨機搜尋。搜尋全程只用 Feb cohort，Mar 只在最後看一次。
tune:
	uv run python scripts/tune.py

mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:
	uv run python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in list(pathlib.Path('.').rglob('__pycache__')) + [pathlib.Path('.pytest_cache'), pathlib.Path('.ruff_cache')]]; print('已清除 __pycache__ / .pytest_cache / .ruff_cache')"

# --- 尚未實作 --------------------------------------------------------------
# 明確報錯而不是安靜地什麼都不做。一個成功但沒有產出的指令會讓人以為跑過了。

eval:
	@echo "make eval 尚未實作 —— M4（機率校準與業務指標）。M3 的評估請用 make compare / select / tune" && exit 1

serve:
	@echo "make serve 尚未實作 —— M6（FastAPI /predict）" && exit 1
