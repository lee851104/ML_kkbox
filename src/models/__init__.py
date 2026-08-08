"""模型訓練與推論。

對應 SPEC §7 的 `src/models/`。
"""

from src.models.train import TrainResult, load_model_config, train_baseline

__all__ = ["TrainResult", "load_model_config", "train_baseline"]
