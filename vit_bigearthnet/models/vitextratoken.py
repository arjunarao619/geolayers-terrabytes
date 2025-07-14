import torch
import torch.nn as nn
import timm

class ViTWithExtraToken(nn.Module):
    def __init__(self, model_name='vit_small_patch16_384', extra_token_dim=384, num_classes=19):
        """
        model_name: base ViT model from timm, e.g., vit_small_patch16_384 for 384-dim embeddings.
        extra_token_dim: dimensionality of the extra token, should match model embedding dim.
        num_classes: number of target classes for classification.
        """
        super(ViTWithExtraToken, self).__init__()
        # Load a ViT model pretrained or not as needed.
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        
        # Initialize an extra token as a learnable parameter.
        self.extra_token = nn.Parameter(torch.zeros(1, 1, extra_token_dim))
        nn.init.trunc_normal_(self.extra_token, std=0.02)
        
        # Extend positional embeddings to account for the new token.
        old_pos_embed = self.backbone.pos_embed  # shape [1, num_patches+1, dim]
        num_extra_tokens = 1  # for our additional token
        new_num_tokens = old_pos_embed.shape[1] + num_extra_tokens

        # Create new positional embeddings with extended sequence length.
        new_pos_embed = torch.zeros(1, new_num_tokens, extra_token_dim)
        # Copy existing positional embeddings.
        new_pos_embed[:, :old_pos_embed.shape[1], :] = old_pos_embed.data
        # Initialize new positional embeddings for the extra token.
        nn.init.trunc_normal_(new_pos_embed[:, old_pos_embed.shape[1]:, :], std=0.02)
        self.backbone.pos_embed = nn.Parameter(new_pos_embed)

    def forward(self, x, extra_token_input=None):
        """
        x: input image tensor of shape [B, C, H, W]
        extra_token_input: optional external embedding of shape [B, dim] to use instead of the learnable token.
        """
        B = x.shape[0]
        # Patch embedding step.
        x = self.backbone.patch_embed(x)  # shape [B, num_patches, dim]
        
        # Prepare class token.
        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)  # shape [B, 1, dim]
        
        # Select extra token: use provided external embedding if given, otherwise use learnable token.
        if extra_token_input is not None:
            # Project external embedding to shape [B, 1, dim] if necessary.
            extra = extra_token_input.unsqueeze(1)  # assumes input shape [B, dim]
        else:
            extra = self.extra_token.expand(B, -1, -1)  # shape [B, 1, dim]
        
        # Concatenate tokens: [CLS] + patch tokens + extra token.
        x = torch.cat((cls_tokens, x, extra), dim=1)  # shape [B, 1 + num_patches + 1, dim]
        
        # Add positional encoding.
        x = x + self.backbone.pos_embed
        
        # Pass through Transformer blocks.
        x = self.backbone.pos_drop(x)  # apply dropout if present
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)
        
        # Classification head applied on CLS token representation.
        cls_out = x[:, 0]  # use the output of the CLS token for classification
        out = self.backbone.head(cls_out)
        return out
