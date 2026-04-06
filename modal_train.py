"""
Modal training script for Parameter Golf.

Usage:
  # First run: download data (run once, persisted in a volume)
  modal run modal_train.py --download

  # Subsequent runs: train directly
  modal run modal_train.py

  # Custom run ID and longer wall-clock
  modal run modal_train.py --run-id my_exp --max-wallclock-seconds 0
"""

import os
import subprocess
from typing import Optional

import modal

# ---------------------------------------------------------------------------
# App + persistent storage
# ---------------------------------------------------------------------------

app = modal.App("parameter-golf")

# Data is persisted across runs so you only download once.
data_vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)
DATA_DIR = "/workspace/data"  # volume mount point (kept separate from code)

# ---------------------------------------------------------------------------
# Container image
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

# Local code is baked into the image layer; Modal caches this layer so only
# changes to the repo trigger a rebuild. Large data dirs are excluded.
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
    volumes={DATA_DIR: data_vol},
    timeout=7200,  # large download can take a while
)
def download_data(variant: str = "sp1024", train_shards: int = 10) -> None:
    """Download FineWeb shards + tokenizer to the persistent volume."""
    import os
    import shutil
    from pathlib import Path

    # The download script writes relative to its own location (data/).
    # We symlink that subtree into the volume so files land on persistent storage.
    code_data = Path("/workspace/parameter-golf/data")
    vol_datasets = Path(DATA_DIR) / "datasets"
    vol_tokenizers = Path(DATA_DIR) / "tokenizers"
    vol_datasets.mkdir(parents=True, exist_ok=True)
    vol_tokenizers.mkdir(parents=True, exist_ok=True)

    # Point the script's output dirs at the volume via symlinks.
    for name, vol_path in [("datasets", vol_datasets), ("tokenizers", vol_tokenizers)]:
        link = code_data / name
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            shutil.rmtree(link)
        link.symlink_to(vol_path)

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

    data_vol.commit()
    print(f"Data downloaded to volume under {DATA_DIR}.")


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100",
    volumes={DATA_DIR: data_vol},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=7200,
)
def train(
    run_id: str = "a100_run",
    variant: str = "sp1024",
    vocab_size: int = 1024,
    max_wallclock_seconds: float = 0.0,
    extra_env: Optional[dict] = None,
) -> None:
    """Run train_gpt_sota.py on a single A100."""
    env = {
        **os.environ,
        "RUN_ID": run_id,
        "DATA_PATH": f"{DATA_DIR}/datasets/fineweb10B_{variant}",
        "TOKENIZER_PATH": f"{DATA_DIR}/tokenizers/fineweb_1024_bpe.model",
        "VOCAB_SIZE": str(vocab_size),
        "MAX_WALLCLOCK_SECONDS": str(max_wallclock_seconds),
        # ~1 epoch through 10 shards (~1.18B tokens at 786K tokens/step)
        "ITERATIONS": "1500",
        "WARMDOWN_ITERS": "300",
        "VAL_LOSS_EVERY": "500",
        # #1 leaderboard config (abaybektursun)
        "NUM_LAYERS": "11",
        "BIGRAM_VOCAB_SIZE": "1536",
        "XSA_LAST_N": "4",
        "SWA_ENABLED": "1",
        "SWA_EVERY": "50",
        "ROPE_DIMS": "16",
        "LN_SCALE": "1",
        "LATE_QAT_THRESHOLD": "0.15",
        "VE_ENABLED": "1",
        "VE_DIM": "128",
        "VE_LAYERS": "9,10",
        "TTT_ENABLED": "1",
        "TTT_LR": "0.002",
        "TTT_EPOCHS": "3",
        "TTT_CHUNK_TOKENS": "32768",
        "TTT_FREEZE_BLOCKS": "0",
        "TTT_MOMENTUM": "0.9",
        "TTT_BATCH_SEQS": "32",
        "TTT_GRAD_CLIP": "1.0",
        "MUON_WD": "0.04",
        "ADAM_WD": "0.04",
        "MATRIX_LR": "0.025",
        "SCALAR_LR": "0.025",
        "TIED_EMBED_LR": "0.035",
        "MUON_MOMENTUM": "0.99",
        "MUON_MOMENTUM_WARMUP_START": "0.92",
        "MUON_MOMENTUM_WARMUP_STEPS": "300",
        "EVAL_STRIDE": "64",
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
    download: bool = False,
    train_shards: int = 10,
    variant: str = "sp1024",
    run_id: str = "a100_run",
    max_wallclock_seconds: float = 0.0,
    gpu: str = "A100",
) -> None:
    """
    --download            Download data to the volume first (run once).
    --train-shards N      Number of training shards to download (default 10).
    --variant NAME        Dataset variant: sp1024 (default) or byte260.
    --run-id NAME         Name for this training run (default: a100_run).
    --max-wallclock-seconds N  Wall-clock budget in seconds; 0 = unlimited.
    --gpu GPU             GPU type: A100 (default), H100, A10G, etc.
    """
    if download:
        print(f"Downloading {train_shards} training shard(s) for variant '{variant}'...")
        download_data.remote(variant=variant, train_shards=train_shards)
        return

    print(f"Launching training run '{run_id}' on A100...")
    train.remote(
        run_id=run_id,
        variant=variant,
        max_wallclock_seconds=max_wallclock_seconds,
    )
