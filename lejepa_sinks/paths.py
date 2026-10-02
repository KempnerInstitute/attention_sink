"""
Centralized path defaults for the LeJEPA sink diagnostics (every script also takes these as CLI arguments).
"""

# Pretrained LeJEPA checkpoints (<run_name>_epoch_49.pt, written by src/trainer.py)
MODEL_DIR = "<PATH_TO_MODELS_DIR>"

# ADE20K root (contains images/validation/ADE_val_*.jpg)
ADE20K_DIR = "<PATH_TO_ADE20K>/ADEChallengeData2016"

# Output directory for per-variant diagnostics (relative to lejepa_sinks/)
RESULTS_DIR = "results"
