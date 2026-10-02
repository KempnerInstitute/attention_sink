"""
Train a linear probe on LeJEPA ViT models for ADE20K semantic segmentation.
"""
import argparse
import sys
import torch
import os

from vit_wrapper import load_vit_wrapper
from trainer import run_train
from paths import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Train ViT Probe on ADE20k (LeJEPA)")
    parser.add_argument(
        "--variant", 
        type=str, 
        required=True, 
        choices=["baseline", "registers", "elementwise", "headwise", "elementwise_reg", "headwise_reg"], 
        help="Model variant"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=7000, 
        help="Seed for probe training (initialization, shuffling)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=".", 
        help="Directory to save the classifier"
    )
    parser.add_argument(
        "--backbone_seed",
        type=int,
        default=9000,
        help="Seed used during backbone pretraining (default 9000)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    # Generate output filename
    output_filename = f"{args.variant}_backbone{args.backbone_seed}_classifier_probeseed_{args.seed}.pt"
    classifier_save_path = os.path.join(args.output_dir, output_filename)
    print(f"Saving classifier to: {classifier_save_path}")

    # Load model
    print(f"Loading {args.variant} LeJEPA model...")
    model = load_vit_wrapper(args.variant, backbone_seed=args.backbone_seed, device=str(device))
    model = model.to(device).float()
    
    print(f"Model loaded. Classifier has {sum(p.numel() for p in model.classifier.parameters())} parameters")
    
    # Use a descriptive variant for logging that includes backbone seed
    log_variant = f"{args.variant}_bb{args.backbone_seed}"

    # Train
    run_train(model, device, DATA_DIR, classifier_save_path, args.seed, log_variant)


if __name__ == "__main__":
    main()
