import os
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path
from typing import Type

import lightning as L
import yaml

from deep_uncertainty.models.discrete_regression_nn import DiscreteRegressionNN
from deep_uncertainty.utils.configs import TrainingConfig
from deep_uncertainty.utils.experiment_utils import get_datamodule
from deep_uncertainty.utils.experiment_utils import get_model

# 3/4/26
# PyTorch 2.6 blocks custom classes from loading in checkpoints by default,
# this whitelists them so the checkpoint can be loaded - Added by Sam Liechty 3/4/26
import torch
from posteriors.vi.diag import VIDiagState
from deep_uncertainty.models.backbones import MLP
from deep_uncertainty.enums import OptimizerType, LRSchedulerType, BetaSchedulerType, AcceleratorType, HeadType, DatasetType
from tensordict._reductions import _make_td
from tensordict._td import TensorDict
from tensordict.tensorclass import NonTensorData
from torchopt.transform.scale_by_adam import ScaleByAdamState
from torchopt.base import EmptyState
torch.serialization.add_safe_globals([MLP, OptimizerType, LRSchedulerType, BetaSchedulerType, AcceleratorType, HeadType, DatasetType, VIDiagState, _make_td, TensorDict, NonTensorData, ScaleByAdamState, EmptyState])

def main(log_dir: Path, config_path: Path, chkp_path: Path):

    if not log_dir.exists():
        os.makedirs(log_dir)
    config = TrainingConfig.from_yaml(config_path)

    datamodule = get_datamodule(
        config.dataset_type,
        config.dataset_spec,
        config.batch_size,
    )

    initializer: Type[DiscreteRegressionNN] = get_model(config, return_initializer=True)[1]
    model = initializer.load_from_checkpoint(chkp_path)
    evaluator = L.Trainer(
        accelerator=config.accelerator_type.value,
        enable_model_summary=False,
        logger=False,
        devices=1,
        num_nodes=1,
    )
    metrics = evaluator.test(model=model, datamodule=datamodule)[0]
    with open(log_dir / "test_metrics.yaml", "w") as f:
        yaml.dump(metrics, f)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Directory to log eval metrics in.",
    )
    parser.add_argument("--config-path", type=str, help="Path to config.yaml used to train model.")
    parser.add_argument(
        "--chkp-path", type=str, help="Path to .ckpt where model weights are saved."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        log_dir=Path(args.log_dir),
        config_path=Path(args.config_path),
        chkp_path=Path(args.chkp_path),
    )
