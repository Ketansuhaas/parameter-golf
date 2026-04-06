"""
Modal training script for Parameter Golf.

Usage:
  # First run: download data (run once, persisted in a volume)
  modal run modal_train.py --download

  # Subsequent runs: train directly (1 epoch, leaderboard #1 config)
  modal run modal_train.py

  # Quick smoke test (10 iterations)
  modal run modal_train.py --debug

  # Custom run
  modal run modal_train.py --run-id my_exp --iterations 9000 --warmdown-iters 3500
"""

import os
import subprocess
from typing import Optional

import modal

# ---------------------------------------------------------------------------
# App + persistent storage
# ---------------------------------------------------------------------------

app = modal.App("parameter-golf")

# ---------------------------------------------------------------------------
# Persistent volume — survives across runs (download once, reuse forever).
# Mounted at DATA_DIR inside every container that needs data.
#
#   Modal Volume (cloud)          Container filesystem
#   ─────────────────────         ──────────────────────────────
#   parameter-golf-data/    →     /workspace/data/
#     datasets/                     datasets/fineweb10B_sp1024/*.bin
#     tokenizers/                   tokenizers/fineweb_1024_bpe.model
#     runs/<run_id>/                logs/, final_model.pt, final_model.int6.ptz
# ---------------------------------------------------------------------------
data_vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)
DATA_DIR = "/workspace/data"

# ---------------------------------------------------------------------------
# Container image — built once, cached by Modal.
#
#   Your local repo               Container filesystem
#   ─────────────────             ──────────────────────────────
#   . (this repo)           →     /workspace/parameter-golf/
#     modal_train.py                modal_train.py
#     train_gpt_sota.py             train_gpt_sota.py
#     data/                         data/   (scripts only, no .bin files)
#     ...                           ...
#
#   data/datasets/ and data/tokenizers/ are excluded — those live in the
#   volume above, not in the image.
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
        add_python="3.11",
    )
    .pip_install(
        "numpy",
        "tqdm",
        "huggingface-hub[hf_transfer]",
        "sentencepiece",
        "datasets",
        "tiktoken",
        "kernels",
        "setuptools",
        "typing-extensions==4.15.0",
        "wandb",
        "flash-attn>=2.7.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Bake local repo into the image (excludes large data dirs).
# Modal caches this layer — only rebuilds when repo files change.
image = image.add_local_dir(
    ".",
    remote_path="/workspace/parameter-golf",
    ignore=[
        "data/datasets/**",
        "data/tokenizers/**",
        ".git/**",
        ".venv/**",
        "**/__pycache__/**",
        "**/*.pyc",
    ],
)

# ---------------------------------------------------------------------------
# Download function  (run once to populate the volume)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={DATA_DIR: data_vol},  # mount the persistent volume at /workspace/data
    timeout=7200,
)
def download_data(variant: str = "sp1024", train_shards: int = 10) -> None:
    """Download FineWeb shards + tokenizer to the persistent volume."""
    import shutil
    from pathlib import Path

    # cached_challenge_fineweb.py writes to data/datasets/ and data/tokenizers/
    # relative to its own location (/workspace/parameter-golf/data/).
    # We redirect those dirs to the volume via symlinks so the files land on
    # persistent storage instead of the ephemeral container filesystem.
    #
    #   /workspace/parameter-golf/data/datasets/  →  symlink  →  /workspace/data/datasets/  (volume)
    #   /workspace/parameter-golf/data/tokenizers/ → symlink  →  /workspace/data/tokenizers/ (volume)
    code_data = Path("/workspace/parameter-golf/data")
    vol_datasets = Path(DATA_DIR) / "datasets"
    vol_tokenizers = Path(DATA_DIR) / "tokenizers"
    vol_datasets.mkdir(parents=True, exist_ok=True)
    vol_tokenizers.mkdir(parents=True, exist_ok=True)

    for name, vol_path in [("datasets", vol_datasets), ("tokenizers", vol_tokenizers)]:
        link = code_data / name
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            shutil.rmtree(link)
        link.symlink_to(vol_path)

    # Download shards from HuggingFace → writes into the volume via symlinks above.
    subprocess.run(
        [
            "python3",
            "data/cached_challenge_fineweb.py",
            "--variant", variant,
            "--train-shards", str(train_shards),
        ],
        cwd="/workspace/parameter-golf",
        check=True,
    )

    # Flush writes to the volume so they persist after the container exits.
    data_vol.commit()
    print(f"Data downloaded to volume under {DATA_DIR}.")


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100",
    volumes={DATA_DIR: data_vol},  # volume mounted read-only here; training only reads data
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=7200,  # hard Modal limit — container is killed after 2hrs regardless
)
def train(
    # --- run config ---
    run_id: str = "a100_run",
    variant: str = "sp1024",
    vocab_size: int = 1024,
    max_wallclock_seconds: float = 0.0,
    # --- schedule (~1 epoch through 10 shards) ---
    iterations: int = 1500,
    warmdown_iters: int = 300,
    warmup_steps: int = 20,
    val_loss_every: int = 500,
    train_log_every: int = 500,
    # --- model ---
    num_layers: int = 11,
    model_dim: int = 512,
    num_heads: int = 8,
    num_kv_heads: int = 4,
    mlp_mult: float = 3.0,
    rope_dims: int = 16,
    rope_base: float = 10000.0,
    ln_scale: bool = True,
    logit_softcap: float = 30.0,
    # --- bigram + XSA ---
    bigram_vocab_size: int = 1536,
    bigram_dim: int = 128,
    xsa_last_n: int = 4,
    # --- training batch ---
    train_batch_tokens: int = 786_432,
    train_seq_len: int = 2048,
    eval_stride: int = 64,
    grad_clip_norm: float = 0.3,
    # --- optimizer (Muon) ---
    matrix_lr: float = 0.025,
    scalar_lr: float = 0.025,
    tied_embed_lr: float = 0.035,
    muon_momentum: float = 0.99,
    muon_momentum_warmup_start: float = 0.92,
    muon_momentum_warmup_steps: int = 300,
    muon_wd: float = 0.04,
    muon_beta2: float = 0.95,
    # --- optimizer (Adam, for scalars) ---
    adam_wd: float = 0.04,
    beta1: float = 0.9,
    beta2: float = 0.95,
    # --- SWA (stochastic weight averaging) ---
    swa_enabled: bool = True,
    swa_every: int = 50,
    # --- VE (value embeddings) ---
    ve_enabled: bool = True,
    ve_dim: int = 128,
    ve_layers: str = "9,10",
    # --- TTT (test-time training) ---
    ttt_enabled: bool = True,
    ttt_lr: float = 0.002,
    ttt_epochs: int = 3,
    ttt_chunk_tokens: int = 32768,
    ttt_freeze_blocks: int = 0,
    ttt_momentum: float = 0.9,
    ttt_batch_seqs: int = 32,
    ttt_grad_clip: float = 1.0,
    # --- QAT ---
    late_qat_threshold: float = 0.15,
    # --- extra overrides (applied last, highest priority) ---
    extra_env: Optional[dict] = None,
) -> None:
    """Run train_gpt_sota.py on a single A100."""
    env = {
        **os.environ,
        # paths
        "RUN_ID": run_id,
        "OUTPUT_DIR": f"{DATA_DIR}/runs/{run_id}",  # logs + checkpoints → volume
        "DATA_PATH": f"{DATA_DIR}/datasets/fineweb10B_{variant}",
        "TOKENIZER_PATH": f"{DATA_DIR}/tokenizers/fineweb_1024_bpe.model",
        "VOCAB_SIZE": str(vocab_size),
        "MAX_WALLCLOCK_SECONDS": str(max_wallclock_seconds),
        # schedule
        "ITERATIONS": str(iterations),
        "WARMDOWN_ITERS": str(warmdown_iters),
        "WARMUP_STEPS": str(warmup_steps),
        "VAL_LOSS_EVERY": str(val_loss_every),
        "TRAIN_LOG_EVERY": str(train_log_every),
        # model
        "NUM_LAYERS": str(num_layers),
        "MODEL_DIM": str(model_dim),
        "NUM_HEADS": str(num_heads),
        "NUM_KV_HEADS": str(num_kv_heads),
        "MLP_MULT": str(mlp_mult),
        "ROPE_DIMS": str(rope_dims),
        "ROPE_BASE": str(rope_base),
        "LN_SCALE": str(int(ln_scale)),
        "LOGIT_SOFTCAP": str(logit_softcap),
        # bigram + XSA
        "BIGRAM_VOCAB_SIZE": str(bigram_vocab_size),
        "BIGRAM_DIM": str(bigram_dim),
        "XSA_LAST_N": str(xsa_last_n),
        # training batch
        "TRAIN_BATCH_TOKENS": str(train_batch_tokens),
        "TRAIN_SEQ_LEN": str(train_seq_len),
        "EVAL_STRIDE": str(eval_stride),
        "GRAD_CLIP_NORM": str(grad_clip_norm),
        # Muon
        "MATRIX_LR": str(matrix_lr),
        "SCALAR_LR": str(scalar_lr),
        "TIED_EMBED_LR": str(tied_embed_lr),
        "MUON_MOMENTUM": str(muon_momentum),
        "MUON_MOMENTUM_WARMUP_START": str(muon_momentum_warmup_start),
        "MUON_MOMENTUM_WARMUP_STEPS": str(muon_momentum_warmup_steps),
        "MUON_WD": str(muon_wd),
        "MUON_BETA2": str(muon_beta2),
        # Adam
        "ADAM_WD": str(adam_wd),
        "BETA1": str(beta1),
        "BETA2": str(beta2),
        # SWA
        "SWA_ENABLED": str(int(swa_enabled)),
        "SWA_EVERY": str(swa_every),
        # VE
        "VE_ENABLED": str(int(ve_enabled)),
        "VE_DIM": str(ve_dim),
        "VE_LAYERS": ve_layers,
        # TTT
        "TTT_ENABLED": str(int(ttt_enabled)),
        "TTT_LR": str(ttt_lr),
        "TTT_EPOCHS": str(ttt_epochs),
        "TTT_CHUNK_TOKENS": str(ttt_chunk_tokens),
        "TTT_FREEZE_BLOCKS": str(ttt_freeze_blocks),
        "TTT_MOMENTUM": str(ttt_momentum),
        "TTT_BATCH_SEQS": str(ttt_batch_seqs),
        "TTT_GRAD_CLIP": str(ttt_grad_clip),
        # QAT
        "LATE_QAT_THRESHOLD": str(late_qat_threshold),
        **(extra_env or {}),
    }

    subprocess.run(
        ["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt_sota.py"],
        cwd="/workspace/parameter-golf",
        env=env,
        check=True,
    )


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    # --- data ---
    download: bool = False,
    train_shards: int = 10,
    variant: str = "sp1024",
    # --- run config ---
    run_id: str = "a100_run",
    max_wallclock_seconds: float = 0.0,
    debug: bool = False,
    # --- schedule ---
    iterations: int = 1500,
    warmdown_iters: int = 300,
    warmup_steps: int = 20,
    val_loss_every: int = 500,
    train_log_every: int = 500,
    # --- model ---
    num_layers: int = 11,
    model_dim: int = 512,
    num_heads: int = 8,
    num_kv_heads: int = 4,
    mlp_mult: float = 3.0,
    rope_dims: int = 16,
    bigram_vocab_size: int = 1536,
    xsa_last_n: int = 4,
    # --- training batch ---
    train_batch_tokens: int = 786_432,
    train_seq_len: int = 2048,
    grad_clip_norm: float = 0.3,
    # --- optimizer ---
    matrix_lr: float = 0.025,
    scalar_lr: float = 0.025,
    tied_embed_lr: float = 0.035,
    muon_momentum: float = 0.99,
    muon_momentum_warmup_steps: int = 300,
    muon_wd: float = 0.04,
    adam_wd: float = 0.04,
    # --- features ---
    swa_enabled: bool = True,
    ve_enabled: bool = True,
    ttt_enabled: bool = True,
    ttt_lr: float = 0.002,
    ttt_epochs: int = 3,
) -> None:
    """
    --download                    Download data to the volume (run once).
    --train-shards N              Shards to download (default 10).
    --variant NAME                Dataset variant: sp1024 or byte260.
    --run-id NAME                 Run name (default: a100_run).
    --max-wallclock-seconds N     Time cap in seconds; 0 = unlimited.
    --debug                       Smoke test: 10 iterations, val every 5.
    --iterations N                Training steps (default 1500 ≈ 1 epoch).
    --warmdown-iters N            LR warmdown steps (default 300).
    --warmup-steps N              LR warmup steps (default 20).
    --val-loss-every N            Validate every N steps (default 500).
    --num-layers N                Transformer layers (default 11).
    --model-dim N                 Model width (default 512).
    --bigram-vocab-size N         Bigram table size (default 1536).
    --xsa-last-n N                XSA on last N layers (default 4).
    --train-batch-tokens N        Tokens per step (default 786432).
    --matrix-lr F                 Muon LR for matrix params (default 0.025).
    --scalar-lr F                 Adam LR for scalar params (default 0.025).
    --tied-embed-lr F             LR for tied embeddings (default 0.035).
    --muon-wd F                   Muon weight decay (default 0.04).
    --adam-wd F                   Adam weight decay (default 0.04).
    --swa-enabled / --no-swa-enabled     Toggle SWA (default on).
    --ve-enabled / --no-ve-enabled       Toggle value embeddings (default on).
    --ttt-enabled / --no-ttt-enabled     Toggle TTT (default on).
    --ttt-lr F                    TTT learning rate (default 0.002).
    --ttt-epochs N                TTT inner epochs (default 3).
    """
    if download:
        print(f"Downloading {train_shards} training shard(s) for variant '{variant}'...")
        download_data.remote(variant=variant, train_shards=train_shards)
        return

    extra_env = None
    if debug:
        run_id = f"{run_id}_debug"
        extra_env = {"ITERATIONS": "10", "WARMUP_STEPS": "2", "WARMDOWN_ITERS": "5", "VAL_LOSS_EVERY": "5", "MUON_MOMENTUM_WARMUP_STEPS": "5"}
        print("DEBUG MODE: 10 iterations, val every 5 steps.")

    print(f"Launching training run '{run_id}' on A100...")
    train.remote(
        run_id=run_id,
        variant=variant,
        max_wallclock_seconds=max_wallclock_seconds,
        iterations=iterations,
        warmdown_iters=warmdown_iters,
        warmup_steps=warmup_steps,
        val_loss_every=val_loss_every,
        train_log_every=train_log_every,
        num_layers=num_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        mlp_mult=mlp_mult,
        rope_dims=rope_dims,
        bigram_vocab_size=bigram_vocab_size,
        xsa_last_n=xsa_last_n,
        train_batch_tokens=train_batch_tokens,
        train_seq_len=train_seq_len,
        grad_clip_norm=grad_clip_norm,
        matrix_lr=matrix_lr,
        scalar_lr=scalar_lr,
        tied_embed_lr=tied_embed_lr,
        muon_momentum=muon_momentum,
        muon_momentum_warmup_steps=muon_momentum_warmup_steps,
        muon_wd=muon_wd,
        adam_wd=adam_wd,
        swa_enabled=swa_enabled,
        ve_enabled=ve_enabled,
        ttt_enabled=ttt_enabled,
        ttt_lr=ttt_lr,
        ttt_epochs=ttt_epochs,
        extra_env=extra_env,
    )
