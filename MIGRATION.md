# 換機交接單（備用機 → 主力機）

> 建立於 2026-08-11，最後一次在備用機 **DESKTOP-8MGKFNP** 上工作之後。
> 這份檔案的用途是讓你在新電腦上有個對照的底。**搬完就可以刪掉**
> （`git rm MIGRATION.md` 一個 commit）—— 它不是專案的一部分。

程式碼與文件全部在 Git 上，`origin/feat/multi-cohort-validation` 的
**4789e51** 是備用機的最後一個 commit。下面列的是**不進 Git、必須手動搬**
的東西，以及搬完怎麼確認成功。

---

## 1. 手動搬的四樣（其餘都能重建）

| 從備用機 | 大小 | 放到新電腦哪裡 |
|---|---|---|
| `C:\Users\User\ml_data\artifacts\`（4 個子資料夾） | 8.3 MB | `<data_root>\artifacts\`（即 `D:\ml_data\artifacts\`） |
| `ML_project\mlflow.db` | 1.2 MB | repo 根目錄 |
| `ML_project\mlartifacts\` | 18 MB | repo 根目錄 |
| `ML_project\reports\kaggle\submission_catboost_fixed.csv` | 59.5 MB | `reports\kaggle\`（**這個資料夾 Git 裡沒有，要自己建**） |

四份 artifact 分別是 `catboost_lead0d`（T=0，離線基準，不可上線）、
`catboost_lead7d`（T−7，`configs/serving.yaml` 預設載這份）、
`catboost_fixed`（固定評分日，Kaggle 管線）、
`catboost_fixed_full`（合併 feb+mar 重訓的最終模型，SPEC §7.20）。

**為什麼這四樣要搬而不是重建**：artifact 重建要 3 × 5 分鐘 + `make final`
55 分鐘；MLflow 的 41 個 run 是 08-08 到 08-11 的實驗歷史，**重建不出來**
（手冊硬性規定第 4 條要的就是它，口試被問「這個數字哪來的」要指得出 run）。

---

## 2. 搬完必做的兩個修正

### 2.1 MLflow 存的是絕對路徑

`mlflow.db` 裡 `experiments.artifact_location` 與 `runs.artifact_uri` 都寫著
備用機的路徑。不改的話 UI 開得起來、指標與參數都在（那些存在 DB 裡），
但 artifact 連結會全部失效。**在新電腦的 repo 根目錄執行一次**：

```bash
python -c "import sqlite3,pathlib; old='C:/Users/User/Desktop/ML_project'; new=pathlib.Path.cwd().as_posix(); c=sqlite3.connect('mlflow.db'); c.execute('update experiments set artifact_location=replace(artifact_location,?,?)',(old,new)); c.execute('update runs set artifact_uri=replace(artifact_uri,?,?)',(old,new)); c.commit(); print('rewritten ->',new)"
```

改完 `uv run mlflow ui` 開得起來、artifact 也點得開，就對了。

### 2.2 驗證提交檔沒有在複製中損壞

sha256 記在 Git 裡的 `reports/kaggle_submission.json`（manifest 存在的用途
之一就是這個）：

```bash
python -c "import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('reports/kaggle/submission_catboost_fixed.csv').read_bytes()).hexdigest())"
```

要等於 `db0f18b322896a9e4ae4207d9d1f19aaaa4a65cd20d80b16cda37d6faaf708b2`。

---

## 3. 新電腦上的步驟

```bash
git fetch origin
git checkout feat/multi-cohort-validation     # ⚠️ 見下方陷阱 1
git pull

# configs/paths.yaml 不進 Git。若是全新 clone，從範本建一份指向 D:/ml_data
cp configs/paths.example.yaml configs/paths.yaml   # 然後改 data_root

uv sync
make test
```

**`make test` 應為 `324 passed · 0 skipped`。** 這是搬遷成功的判準：
`pyproject.toml` 與 `uv.lock` 在備用機上查核過完全同步、沒有側裝任何套件，
所以新機 `uv sync` 出來的環境會一致。

零 skip 這件事本身也是 M6 的成果之一 —— 八條紅線到齊，測試清單裡不再有
「還欠什麼」的提醒（SPEC §7.20）。

---

## 4. 三個陷阱

1. **clone 完預設在 `main`，而 `main` 落後工作分支 47 個 commit。**
   所有 M4–M6 的工作都在 `feat/multi-cohort-validation` 上。要不要合回 main
   是還沒做的決定。
2. **原始資料 33 GB 不進 Git。** 新電腦的 `D:\ml_data\raw\` 要有那 10 個檔案
   （`transactions.csv`、`user_logs.csv` 等）。不在的話 `make data-all` 重下載
   8.95 GB 再解壓，操作簡單但要好幾個小時。**interim 快取、名單、提交檔、
   artifact —— 所有「可以重建」的東西都是從 raw 算出來的**，沒有例外。
3. **`kaggle.json` 憑證也不進 Git**（下載資料與送出提交都要它）。

---

## 5. 不要搬的東西

`configs\paths.yaml`（機器專屬：備用機指 C 槽，主力機是 D 槽）、`.venv\`、
`.idea\`、`__pycache__\`、`.pytest_cache\`、`.ruff_cache\`、`catboost_info\`。

`reports\explanations\*.csv`（投放名單與稽核檔）也不必搬 —— SPEC §7.14 明說
那是函式的輸出、不是資產，`make explain` 五分鐘重生。

---

## 6. 可重建的東西與成本

| 東西 | 指令 | 需要 | 成本 |
|---|---|---|---|
| `.venv` | `uv sync` | 網路 | 1–2 分鐘 |
| interim 快取（9.9 GB / 18 個 parquet） | 跑任何 `make` 目標時自動建 | raw | 掃 30 GB 日誌 |
| 投放名單 + 原因碼 CSV | `make explain` | artifacts + interim | 5 分鐘 |
| 16 張圖表 | **不用產，已在 Git 裡** | — | 0 |
| Kaggle 提交檔 | `make kaggle` | artifacts + interim + raw | 5 分鐘 |
| 三份單一 cohort artifact | `make artifact` / `-t7` / `-fixed` | raw + interim | 各 5 分鐘 |
| 合併重訓的最終模型 | `make final` | raw + interim | **55 分鐘** |

最後兩列是複製那 8.3 MB 的理由。

---

## 7. 接下來的待辦（都已寫進 Git，不會遺失）

**M6 只剩兩塊**：

- **Docker** —— 備用機沒裝 Docker，所以那塊只寫得出檔案、驗不了。新電腦若有
  Docker Desktop 就能真的 build 起來驗（這個專案的標準是「沒跑過的不算數」）。
- **HF Spaces Demo**

**兩個待決的判斷**（決定權在你）：

- **要不要用 `catboost_fixed_full` 產生新的 Kaggle 提交檔。** 它是合併兩個
  cohort 重訓的最終模型，理論上資料更多、時間更近。若要做，建議先讓
  manifest 檔名帶 artifact 名，兩份並存 —— 現在的 `reports/kaggle_submission.json`
  記著上一次的**事前登記**（預期 0.17342），覆蓋掉就沒有兩次的對照了。
- **`p*` 要用哪一期的流失率推導。** 合併重訓把它從 0.4534 推到 0.5231
  （+15.4%），原因是訓練 cohort 的流失率從 5.8841% 變成兩期混合的 6.7682%
  —— **投放名單會因此變小，而那與模型的排序能力無關**。細節見 SPEC §9 第 4 項
  與 §7.20 第六點。改這個不必重訓。

**還沒送出的事**：Kaggle late submission。提交檔已產生（907,471 列），
事前登記的預期是 0.17342，分數回來要填進 README 成果表與 SPEC §3.3，
**無論結果如何** —— 那是那條門檻的意義。

---

## 8. 備用機的 artifact 有一個旗標要知道

`catboost_fixed_full` 與 `catboost_fixed` 的 metadata 都記著 `git_dirty: true`
（匯出當下工作區有未提交的改動），所以它們**無法用 git SHA 回溯**。那個旗標
正在做它該做的事，不是 bug。要一份 provenance 乾淨的 artifact，在新電腦上
commit 之後重跑 `make final` 即可（已驗證逐位重現：8 個 fold 分數、最佳輪數
1974、校準器的每個數字兩次執行完全相同）。
