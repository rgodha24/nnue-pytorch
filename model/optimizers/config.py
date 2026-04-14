from dataclasses import dataclass
from typing import Literal

from .ranger22_wrapper import Ranger22Config, Ranger22Wrapper
from .schedulefree_wrapper import ScheduleFreeConfig, ScheduleFreeWrapper


@dataclass(kw_only=True)
class OptimizerConfig(Ranger22Config, ScheduleFreeConfig):
    optimizer_name: Literal["schedulefree", "ranger22"] = "ranger22"
    """Which optimizer to use. """

    ft_weight_decay: float = 0.0
    """Weight decay to apply to the feature transformer parameters."""

    dense_weight_decay: float = 0.0
    """Weight decay to apply to the dense layer parameters."""

    lr: float = 8.75e-4
    """Initial learning rate."""

    def get_optimizer_wrapper(self, max_epoch, num_batches_per_epoch):
        optimizer_name = self.optimizer_name.lower().strip()
        if optimizer_name == "schedulefree":
            wrapper = ScheduleFreeWrapper(self)
        elif optimizer_name == "ranger22":
            wrapper = Ranger22Wrapper(self, max_epoch, num_batches_per_epoch)
        else:
            raise ValueError(
                f"Unknown optimizer_name: '{optimizer_name}'. Expected 'schedulefree' or 'ranger22'."
            )

        if self.dense_weight_decay > 0.0 or self.ft_weight_decay > 0.0:
            print(
                f"Using weight decay - ft_weight_decay: {self.ft_weight_decay}, dense_weight_decay: {self.dense_weight_decay}"
            )
        return wrapper
