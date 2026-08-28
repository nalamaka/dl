from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    data_dir: Path = ROOT_DIR / "data" / "sample-submission-version"

    train_zh_path: Path = data_dir / "TM-training-set" / "chinese.txt"
    train_en_path: Path = data_dir / "TM-training-set" / "english.txt"
    test_zh_path: Path = data_dir / "Test-set" / "Niu.test.txt"
    test_reference_path: Path = data_dir / "Reference-for-evaluation" / "Niu.test.reference"

    save_dir: Path = Path(__file__).resolve().parent / "checkpoints"
    vocab_dir: Path = Path(__file__).resolve().parent / "vocabs"
    tokenizer_dir: Path = Path(__file__).resolve().parent / "tokenizers"
    log_dir: Path = Path(__file__).resolve().parent / "logs"

    seed: int = 42
    val_ratio: float = 0.02
    max_train_samples: int | None = None

    min_freq: int = 2
    max_src_vocab_size: int = 60000
    max_tgt_vocab_size: int = 60000
    max_src_len: int = 128
    max_tgt_len: int = 128

    batch_size: int = 64
    num_workers: int = 0
    epochs: int = 12
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    use_amp: bool = True
    save_every_steps: int = 0

    scheduler: str = "warmup_cosine"  # ["none", "warmup_cosine"]
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1

    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    tie_embeddings: bool = True

    max_decode_len: int = 128
    log_interval: int = 100
    log_every_steps: int = 10
    device: str = "cuda"
    clear_cuda_cache_every: int = 1
    preview_samples: int = 0
    preview_max_decode_len: int = 64
    eval_batch_size: int = 4
    test_batch_size: int = 4
    max_eval_samples: int = 100
    max_test_samples: int = 1000
    eval_decode_len: int = 48
    test_decode_len: int = 48
    decode_strategy: str = "greedy"  # ["greedy", "beam"]
    beam_size: int = 4
    length_penalty: float = 0.6
    no_repeat_ngram_size: int = 3
