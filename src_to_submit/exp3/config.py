from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data_path: Path = Path("../../data/tang.npz")

    # 16GB-friendly defaults for training from scratch (no LLM fine-tuning).
    batch_size: int = 16
    grad_accum_steps: int = 2

    d_model: int = 384
    nhead: int = 6
    num_layers: int = 6
    dim_feedforward: int = 1536
    dropout: float = 0.1
    max_seq_len: int = 128
    max_line_length: int = 12

    # Cyclic position embedding for quatrain-like structure (line_length + punctuation slot).
    pattern_cycle: int = 8

    # More conservative optimization defaults for stability.
    # Lower base LR + longer warmup reduce sudden loss spikes.
    lr: float = 6e-5
    weight_decay: float = 0.01
    epochs: int = 30
    max_gen_len: int = 128

    warmup_ratio: float = 0.12
    min_lr_ratio: float = 0.1
    max_grad_norm: float = 0.3
    scheduler_type: str = "cosine"  # ["cosine", "linear", "none"]
    use_amp: bool = True
    log_interval: int = 200

    eval_samples: int = 20

    save_dir: Path = Path("./checkpoints")
    device: str = "cuda"
