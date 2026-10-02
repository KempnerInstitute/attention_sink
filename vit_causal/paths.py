"""
Centralized path definitions for the ViT causal-intervention experiments.

Edit the placeholders below, or override them with environment variables.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# ImageNet-1k validation split in ImageFolder layout (1000 class folders, 50,000 images)
IMAGENET_VAL_DIR = os.environ.get("ATTN_SINK_IMAGENET_VAL", "<PATH_TO_IMAGENET_VAL>")

# Where all per-model outputs (results/<model>/...) and the analysis tables are written
RESULTS_DIR = os.environ.get("ATTN_SINK_VIT_CAUSAL_RESULTS", os.path.join(_HERE, "results"))

# DINOv2 code and weights are fetched with torch.hub and cached under $TORCH_HOME/hub (default ~/.cache/torch/hub).
