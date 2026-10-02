"""
Wrapper for LeJEPA ViT models for Pascal VOC 2012 segmentation probing.
Takes in images, outputs segmentation logits.

LeJEPA checkpoints save the full ViTEncoder (backbone + projector) state dict
under the 'model' key. We extract the 'backbone.*' keys to load only the
VisionTransformer.
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from vit import VisionTransformer
from paths import MODEL_DIR


def load_config_from_checkpoint(checkpoint_path: str) -> dict:
    """Load config JSON that corresponds to a checkpoint file."""
    basename = os.path.basename(checkpoint_path)
    # Remove 'last_' or 'best_' prefix
    config_name = basename.replace("best_", "").replace("last_", "").replace(".pt", "_config.json")
    config_path = os.path.join(os.path.dirname(checkpoint_path), config_name)
    
    with open(config_path, "r") as f:
        return json.load(f)


def extract_backbone_state_dict(checkpoint_path: str, device: torch.device) -> dict:
    """
    Load a LeJEPA checkpoint and extract the backbone (VisionTransformer) weights.
    
    LeJEPA saves the full ViTEncoder state dict (backbone.* + proj.*) under the
    'model' key. We strip the 'backbone.' prefix to get a clean VisionTransformer
    state dict.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" in checkpoint:
        full_state = checkpoint["model"]
    else:
        full_state = checkpoint
    
    # Extract only backbone.* keys and strip the prefix
    backbone_state = {}
    for k, v in full_state.items():
        if k.startswith("backbone."):
            backbone_state[k[len("backbone."):]] = v
    
    if len(backbone_state) == 0:
        # Fallback: maybe the state dict is already just the backbone
        backbone_state = full_state
    
    return backbone_state


class ViTSegmentationWrapper(nn.Module):
    """
    Wraps a pretrained LeJEPA ViT for semantic segmentation probing.
    
    The backbone is frozen; only the linear classifier head is trained.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int = 20,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.device = torch.device(device)
        
        # Load config from JSON
        config = load_config_from_checkpoint(checkpoint_path)
        
        self.num_register_tokens = config.get("num_register_tokens", 0)
        embed_dim = config.get("embed_dim", 1024)
        
        # Build ViT from config (LeJEPA ViT has no pool_type)
        self.vit = VisionTransformer(
            img_size=config.get("img_size", 224),
            patch_size=config.get("patch_size", 16),
            in_chans=3,
            embed_dim=embed_dim,
            depth=config.get("depth", 24),
            num_heads=config.get("num_heads", 16),
            mlp_ratio=config.get("mlp_ratio", 4.0),
            qkv_bias=config.get("qkv_bias", False),
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            num_register_tokens=self.num_register_tokens,
            gate_type=config.get("gate_type"),
        )
        
        # Load backbone weights from LeJEPA checkpoint
        backbone_state = extract_backbone_state_dict(checkpoint_path, self.device)
        self.vit.load_state_dict(backbone_state, strict=True)
        self.vit.eval()
        
        # Freeze backbone
        for p in self.vit.parameters():
            p.requires_grad_(False)
        
        # Linear probe classifier: maps patch embeddings to class logits
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for segmentation.
        
        Args:
            x: Input images (B, 3, H, W). Should be multiples of patch_size (16).
               
        Returns:
            logits: (B, num_classes, H', W') where H'=H/16, W'=W/16
        """
        B = x.shape[0]
        
        # Patch embed
        x = self.vit.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.vit.cls_token.expand(B, -1, -1)
        if self.vit.register_tokens is not None:
            reg_tokens = self.vit.register_tokens.expand(B, -1, -1)
            x = torch.cat((cls_tokens, reg_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding (need to interpolate if image size differs from training)
        x = x + self._interpolate_pos_embed(x.shape[1])
        x = self.vit.pos_drop(x)
        
        # Forward through blocks
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)
        
        # Extract patch tokens (skip CLS and register tokens)
        num_prefix = 1 + self.num_register_tokens
        patch_tokens = x[:, num_prefix:]  # (B, num_patches, embed_dim)
        
        # Apply classifier to each patch
        logits = self.classifier(patch_tokens)  # (B, num_patches, num_classes)
        
        # Reshape to spatial grid
        H = W = int(patch_tokens.shape[1] ** 0.5)
        logits = logits.transpose(1, 2)  # (B, num_classes, num_patches)
        logits = logits.reshape(B, -1, H, W)  # (B, num_classes, H, W)
        
        return logits
    
    def _interpolate_pos_embed(self, num_tokens: int) -> torch.Tensor:
        """Interpolate positional embeddings if needed."""
        pos_embed = self.vit.pos_embed
        
        if pos_embed.shape[1] == num_tokens:
            return pos_embed
        
        # Need to interpolate
        num_prefix = 1 + self.num_register_tokens
        
        # Separate prefix (CLS + registers) and patch embeddings
        prefix_embed = pos_embed[:, :num_prefix]
        patch_embed = pos_embed[:, num_prefix:]
        
        # Original patch grid size
        orig_size = int(patch_embed.shape[1] ** 0.5)
        
        # Target patch grid size
        target_num_patches = num_tokens - num_prefix
        target_size = int(target_num_patches ** 0.5)
        
        # Reshape and interpolate
        patch_embed = patch_embed.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
        patch_embed = F.interpolate(patch_embed, size=(target_size, target_size), mode='bicubic', align_corners=False)
        patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, target_size * target_size, -1)
        
        return torch.cat([prefix_embed, patch_embed], dim=1)


def load_vit_wrapper(variant: str, backbone_seed: int = 9000, device: str = "cuda") -> ViTSegmentationWrapper:
    """
    Load a ViT wrapper for a specific LeJEPA variant.

    Args:
        variant: One of 'baseline', 'registers', 'elementwise', 'headwise',
                 'elementwise_reg', 'headwise_reg'
        backbone_seed: Seed used during backbone pretraining (default 9000)
        device: Device to load model on

    Returns:
        ViTSegmentationWrapper with frozen backbone
    """
    seed = backbone_seed
    checkpoints = {
        "baseline": f"{MODEL_DIR}/last_ViT-L_baseline_v4_pd128_lam0.02_seed{seed}_lr0.0005_wd0.05_bs64_ep100.pt",
        "registers": f"{MODEL_DIR}/last_ViT-L_reg4_v4_pd128_lam0.02_seed{seed}_lr0.0005_wd0.05_bs64_ep100.pt",
        "elementwise": f"{MODEL_DIR}/last_ViT-L_elementwise_v4_pd128_lam0.02_seed{seed}_lr0.0005_wd0.05_bs64_ep100.pt",
        "headwise": f"{MODEL_DIR}/last_ViT-L_headwise_v4_pd128_lam0.02_seed{seed}_lr0.0005_wd0.05_bs64_ep100.pt",
        "elementwise_reg": f"{MODEL_DIR}/last_ViT-L_elementwise_reg4_v4_pd128_lam0.02_seed{seed}_lr0.0005_wd0.05_bs64_ep100.pt",
        "headwise_reg": f"{MODEL_DIR}/last_ViT-L_headwise_reg4_v4_pd128_lam0.02_seed{seed}_lr0.0005_wd0.05_bs64_ep100.pt",
    }

    if variant not in checkpoints:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(checkpoints.keys())}")

    return ViTSegmentationWrapper(
        checkpoint_path=checkpoints[variant],
        num_classes=20,
        device=device,
    )
