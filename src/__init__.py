"""KKBox 訂閱流失預測與挽回決策 —— 專案程式碼。

目錄對應 SPEC §7 的 repo 結構。目前只建了「現在真的有東西要放」的層：

    config.py       路徑與設定，跨層共用
    data/           資料載入與 as-of 切分（SPEC §4.3、紅線 1、2）
    evaluation/     指標與分群回報（紅線 8、SPEC §4.5）

SPEC 還列了 features/、models/、serving/，等 M2、M1、M6 要用時再建。
空目錄是雜訊，不先開。
"""
