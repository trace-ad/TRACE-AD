"""CLIP patch encoding and phase-grid resampling."""
import os

import numpy as np
import torch
import torch.nn.functional as F

from .render import preprocess

# Half-precision toggle for CLIP vision inference (env CLIP_DTYPE).
# fp32 (default) is byte-for-byte unchanged; fp16/bf16 cast the vision model
# and pixel_values, but returned features are always float32 numpy so that
# downstream scoring is unchanged in dtype.
_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def clip_dtype():
    return _DTYPES.get(os.environ.get("CLIP_DTYPE", "fp32").lower(), torch.float32)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_clip(model_name, device):
    from transformers import CLIPModel
    model = CLIPModel.from_pretrained(model_name).eval().to(device)
    dt = clip_dtype()
    if dt != torch.float32:
        model = model.to(dt)
    return model


@torch.no_grad()
def encode(images, model, device, batch_size=128):
    """Encode PIL images to (patch_grid[14,14,D], gray224) tuples.

    Patch tokens are projected into CLIP joint space and L2-normalized.
    """
    out = []
    mdtype = next(model.parameters()).dtype
    for i in range(0, len(images), batch_size):
        ch = images[i:i + batch_size]
        px = preprocess(ch, device)
        if px.dtype != mdtype:
            px = px.to(mdtype)
        pat = model.visual_projection(
            model.vision_model(pixel_values=px).last_hidden_state[:, 1:, :])
        pat = pat / pat.norm(dim=-1, keepdim=True)
        p14 = pat.reshape(len(ch), 14, 14, -1).float().cpu().numpy()
        if device == "cuda":
            torch.cuda.synchronize()
        for b, im in enumerate(ch):
            out.append((p14[b], np.asarray(im.convert("L").resize((224, 224)), float)))
    return out


def grid_columns(p14, gray, grid_cols):
    """Resample the 14x14 patch grid to ``grid_cols`` columns.

    Returns ``(g, fg)`` where ``g`` is [grid_cols, grid_cols, D] L2-normalized
    features and ``fg`` is the [grid_cols, grid_cols] foreground (ink) mask.
    """
    t = torch.from_numpy(p14).permute(2, 0, 1)[None]
    g = F.interpolate(t, size=grid_cols, mode="bilinear", align_corners=True)[0]
    g = g.permute(1, 2, 0).numpy()
    g = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-9)
    P = 224 // grid_cols
    fg = (gray.reshape(grid_cols, P, grid_cols, P).min((1, 3)) < 110.0)
    return g, fg
