"""
Centralized path definitions for the cached-feature segmentation probes (Table 1).
"""
import os

# Base project directory
PRJ_DIR = "<PATH_TO_PROJECT_ROOT>"

# LeJEPA checkpoints written by src/trainer.py (<run_name>_epoch_49.pt)
MODEL_DIR = "<PATH_TO_MODELS_DIR>"

# ADE20K SceneParse150: <root>/images/{training,validation}/*.jpg, <root>/annotations/{training,validation}/*.png
ADE20K_DIR = "<PATH_TO_ADE20K>/ADEChallengeData2016"

# Pascal VOC 2012 trainval, official tar layout: <root>/{JPEGImages,SegmentationClass,ImageSets/Segmentation}
VOC2012_DIR = "<PATH_TO_VOC2012>/VOCdevkit/VOC2012"

# Cached backbone features: <FEATURES_DIR>/<cell>_ep<E>/<dataset>_<split>/{feats.npy,labels.npy,meta.json}
# (per model: ADE20K train 42 GB + val 4 GB, VOC 3 GB + 3 GB)
FEATURES_DIR = f"{PRJ_DIR}/cached_features"

# Probe results: <RESULTS_DIR>/<dataset>/<cell>_ep<E>_seed<S>[_probe<N>ep]/result.json
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
