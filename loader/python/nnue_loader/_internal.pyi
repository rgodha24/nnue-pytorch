from collections.abc import Iterator, Sequence
from typing import TypedDict

import numpy as np
import numpy.typing as npt

class Batch(TypedDict):
    num_inputs: int
    size: int
    num_active_white_features: int
    num_active_black_features: int
    max_active_features: int
    is_white: npt.NDArray[np.float32]
    outcome: npt.NDArray[np.float32]
    score: npt.NDArray[np.float32]
    white: npt.NDArray[np.int32]
    black: npt.NDArray[np.int32]
    psqt_indices: npt.NDArray[np.int32]
    layer_stack_indices: npt.NDArray[np.int32]

class PipelineStats(TypedDict):
    decoded_entries: int
    encoded_entries: int
    skipped_entries: int
    produced_batches: int
    position_queue_len: int
    ready_slabs_len: int
    free_slabs_len: int

class BatchStream(Iterator[Batch]):
    def __init__(
        self,
        feature_set: str,
        filenames: Sequence[str],
        batch_size: int,
        *,
        encoding_threads: int | None = None,
        slab_count: int | None = None,
        position_queue_capacity: int | None = None,
        position_queue_high_watermark: int | None = None,
        position_queue_low_watermark: int | None = None,
        shuffle_buffer_entries: int = 1_000_000,
        shuffle_chunks: bool = True,
        cyclic: bool = False,
        seed: int | None = None,
        filtered: bool = False,
        random_fen_skipping: int = 0,
        wld_filtered: bool = False,
        early_fen_skipping: int = -1,
        simple_eval_skipping: int = 0,
        param_index: int = 0,
        pc_y1: float = 1.0,
        pc_y2: float = 2.0,
        pc_y3: float = 1.0,
    ) -> None: ...
    def __iter__(self) -> BatchStream: ...
    def __next__(self) -> Batch: ...
    def next_batch(self) -> Batch | None: ...
    def stats(self) -> PipelineStats: ...
    def close(self) -> None: ...

def create_batch_stream(
    feature_set: str,
    filenames: Sequence[str],
    batch_size: int,
    *,
    encoding_threads: int | None = None,
    slab_count: int | None = None,
    position_queue_capacity: int | None = None,
    position_queue_high_watermark: int | None = None,
    position_queue_low_watermark: int | None = None,
    shuffle_buffer_entries: int = 1_000_000,
    shuffle_chunks: bool = True,
    cyclic: bool = False,
    seed: int | None = None,
    filtered: bool = False,
    random_fen_skipping: int = 0,
    wld_filtered: bool = False,
    early_fen_skipping: int = -1,
    simple_eval_skipping: int = 0,
    param_index: int = 0,
    pc_y1: float = 1.0,
    pc_y2: float = 2.0,
    pc_y3: float = 1.0,
) -> BatchStream: ...
def hello() -> str: ...
