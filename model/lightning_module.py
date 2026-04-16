import collections

import lightning as L
import torch
from torch import Tensor, nn
from torchmetrics import MeanMetric, MetricCollection

from .config import NNUELightningConfig
from .model import NNUEModel
from .quantize import QuantizationConfig


def _get_parameters(layers: list[nn.Module], get_biases: bool = False):
    return [
        p
        for layer in layers
        for name, p in layer.named_parameters()
        if ("bias" in name) == get_biases and p.requires_grad
    ]


class NNUE(L.LightningModule):
    """
    lambda_ = 0.0 - purely based on game results
    0.0 < lambda_ < 1.0 - interpolated score and result
    lambda_ = 1.0 - purely based on search scores
    """

    def __init__(
        self,
        config: NNUELightningConfig,
        max_epoch=None,
        num_batches_per_epoch=None,
        quantize_config=QuantizationConfig(),
        param_index=0,
        num_psqt_buckets=8,
        num_ls_buckets=8,
    ):
        super().__init__()

        self.model: NNUEModel = NNUEModel(
            config.features,
            config.model_config,
            quantize_config,
            num_psqt_buckets,
            num_ls_buckets,
        )
        self.config = config
        self.max_epoch = max_epoch
        self.num_batches_per_epoch = num_batches_per_epoch
        self.param_index = param_index

        # lazy init so `resume_from_model` with config changes works correctly
        self.optimizer_wrapper = None

        self.loss_metrics = MetricCollection ({
            "train_loss_epoch": MeanMetric(),
            "val_loss_epoch": MeanMetric(),
            "test_loss_epoch": MeanMetric(),
        })

        # ---- H2D overlap plumbing ----
        # A dedicated CUDA stream for Pinned->Device copies so they can run
        # concurrently with the previous iteration's compute on the default
        # stream (Ranger22, FC/FT backward, etc.). Trace analysis on the
        # 5060 Ti showed the 11.8 ms/step of Memcpy HtoD was being serialized
        # on the compute stream between the optimizer and the next forward;
        # moving it to a side stream fully hides it behind Ranger22's 15 ms.
        #
        # The CPU batch tuple is also retained for a couple of iterations so
        # that the Rust loader's underlying CPU buffers are not recycled
        # while the async copy is still in flight.
        self._xfer_stream: torch.cuda.Stream | None = None
        self._pending_cpu_batches: collections.deque = collections.deque(maxlen=3)

    # --- setup optimizers and training hooks ---
    def configure_optimizers(self):
        if self.max_epoch is None:
            print("[NNUE] Required parameter for training not set: max_epoch")

        optimizer_config = self.config.optimizer_config
        self.optimizer_wrapper = optimizer_config.get_optimizer_wrapper(
            self.max_epoch, self.num_batches_per_epoch
        )

        LR = optimizer_config.lr
        ft_wd = optimizer_config.ft_weight_decay
        dense_wd = optimizer_config.dense_weight_decay

        train_params = [
            # Feature Transformer
            {
                "params": _get_parameters([self.model.input], get_biases=False),
                "lr": LR,
                "weight_decay": ft_wd,
            },
            {
                "params": _get_parameters([self.model.input], get_biases=True),
                "lr": LR,
                "weight_decay": 0.0,
            },
            # Dense Layer Stacks
            {
                "params": [self.model.layer_stacks.l1.factorized_linear.weight],
                "lr": LR,
                "weight_decay": dense_wd,
            },
            {
                "params": [self.model.layer_stacks.l1.factorized_linear.bias],
                "lr": LR,
                "weight_decay": 0.0,
            },
            {
                "params": [self.model.layer_stacks.l1.linear.weight],
                "lr": LR,
                "weight_decay": dense_wd,
            },
            {
                "params": [self.model.layer_stacks.l1.linear.bias],
                "lr": LR,
                "weight_decay": 0.0,
            },
            {
                "params": [self.model.layer_stacks.l2.linear.weight],
                "lr": LR,
                "weight_decay": dense_wd,
            },
            {
                "params": [self.model.layer_stacks.l2.linear.bias],
                "lr": LR,
                "weight_decay": 0.0,
            },
            {
                "params": [self.model.layer_stacks.output.linear.weight],
                "lr": LR,
                "weight_decay": dense_wd,
            },
            {
                "params": [self.model.layer_stacks.output.linear.bias],
                "lr": LR,
                "weight_decay": 0.0,
            },
        ]

        return self.optimizer_wrapper.configure_optimizers(train_params)

    def on_train_epoch_start(self):
        self.optimizer_wrapper.on_train_epoch_start(self)

    def on_train_epoch_end(self):
        self.optimizer_wrapper.on_train_epoch_end(self)
        self._log_epoch_end("train_loss_epoch")

    def on_validation_epoch_start(self):
        self.optimizer_wrapper.on_validation_epoch_start(self)

    def on_validation_epoch_end(self):
        self._log_epoch_end("val_loss_epoch")

    def on_test_epoch_start(self):
        self.optimizer_wrapper.on_test_epoch_start(self)

    def on_test_epoch_end(self):
        self._log_epoch_end("test_loss_epoch")

    def on_save_checkpoint(self, checkpoint):
        self.optimizer_wrapper.on_save_checkpoint(self, checkpoint)

    def on_train_batch_start(self, batch, batch_idx):
        # ``transfer_batch_to_device`` launched the H2D copy on ``_xfer_stream``
        # to overlap with the previous iteration's GPU work. Make the default
        # compute stream wait for that copy before any kernels read the batch,
        # and record_stream each tensor so the caching allocator doesn't free
        # the GPU buffer while compute is still using it.
        if self._xfer_stream is not None:
            compute_stream = torch.cuda.current_stream(self._xfer_stream.device)
            compute_stream.wait_stream(self._xfer_stream)
            for t in batch:
                if isinstance(t, torch.Tensor) and t.is_cuda:
                    t.record_stream(compute_stream)
        self.optimizer_wrapper.on_train_batch_start(self, batch, batch_idx)

    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        # Fast path: CPU training or non-cuda device. Fall back to Lightning's
        # default (which just calls ``.to(device, non_blocking=True)``).
        if not torch.cuda.is_available() or device.type != "cuda":
            return super().transfer_batch_to_device(batch, device, dataloader_idx)

        if self._xfer_stream is None:
            self._xfer_stream = torch.cuda.Stream(device=device)

        xfer_stream = self._xfer_stream
        with torch.cuda.stream(xfer_stream):
            if isinstance(batch, (list, tuple)):
                batch_gpu = type(batch)(
                    t.to(device, non_blocking=True)
                    if isinstance(t, torch.Tensor) and t.device.type == "cpu"
                    else t
                    for t in batch
                )
            elif isinstance(batch, torch.Tensor) and batch.device.type == "cpu":
                batch_gpu = batch.to(device, non_blocking=True)
            else:
                batch_gpu = batch

        # Keep the CPU batch alive for a few iterations so the Rust loader
        # doesn't recycle the underlying pinned buffer before our async copy
        # completes. ``maxlen=3`` gives >3x headroom over the ~12 ms copy.
        self._pending_cpu_batches.append(batch)
        return batch_gpu

    def _log_epoch_end(self, loss_type):
        self.log(
            f"{loss_type}",
            self.loss_metrics[f"{loss_type}"],
            prog_bar=False,
            sync_dist=True,
            on_epoch=True,
            on_step=False,
        )

    # --- Training step implementation ---

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        return self.step_(batch, batch_idx, "train_loss")

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, "val_loss")

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, "test_loss")

    def step_(self, batch: tuple[Tensor, ...], batch_idx, loss_type):
        _ = batch_idx  # unused, but required by pytorch-lightning

        (
            us,
            them,
            white_indices,
            white_values,
            black_indices,
            black_values,
            outcome,
            score,
            psqt_indices,
            layer_stack_indices,
        ) = batch

        scorenet = (
            self.model(
                us,
                them,
                white_indices,
                white_values,
                black_indices,
                black_values,
                psqt_indices,
                layer_stack_indices,
            )
            * self.model.quantization.nnue2score
        )

        p = self.config.loss_params
        # convert the network and search scores to an estimate match result
        # based on the win_rate_model, with scalings and offsets optimized
        q = (scorenet - p.in_offset) / p.in_scaling
        qm = (-scorenet - p.in_offset) / p.in_scaling
        qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())

        s = (score - p.out_offset) / p.out_scaling
        sm = (-score - p.out_offset) / p.out_scaling
        pf = 0.5 * (1.0 + s.sigmoid() - sm.sigmoid())

        # blend that eval based score with the actual game outcome
        t = outcome
        actual_lambda = p.start_lambda + (p.end_lambda - p.start_lambda) * (
            self.current_epoch / self.max_epoch
        )
        pt = pf * actual_lambda + t * (1.0 - actual_lambda)

        # use a MSE-like loss function
        loss = torch.pow(torch.abs(pt - qf), p.pow_exp)
        if p.qp_asymmetry != 0.0:
            loss = loss * ((qf > pt) * p.qp_asymmetry + 1)

        weights = 1 + (2.0**p.w1 - 1) * torch.pow((pf - 0.5) ** 2 * pf * (1 - pf), p.w2)
        loss = (loss * weights).sum() / weights.sum()

        self.loss_metrics[f"{loss_type}_epoch"].update(loss)
        self.log(
            loss_type,
            loss,
            prog_bar=False,
            sync_dist=False,
            on_epoch=False,
            on_step=True,
        )
        return loss
