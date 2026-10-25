"""Rendering: turn a 1-D beat into a CLIP-ready 224x224 image.

Uses a PIL line rasterizer (supersample + LANCZOS) instead of matplotlib, and a
manual CLIP normalization that bypasses ``CLIPProcessor`` -- both verified to be
numerically identical to the matplotlib/CLIPProcessor path while being far
faster.
"""
import numpy as np
import torch
from PIL import Image, ImageDraw

# OpenAI CLIP image normalization constants.
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def znorm(a):
    a = np.asarray(a, float)
    return (a - a.mean()) / (a.std() + 1e-9)


def draw_segment(seg, xhi, yr, supersample=3):
    """Rasterize ``seg`` as a line plot on a 224x224 white canvas.

    Draws at ``supersample``x resolution then downsamples with LANCZOS so the
    line width matches matplotlib's ``lw=1.5pt`` at 100 dpi.
    """
    W = H = 224
    w, h = W * supersample, H * supersample
    im = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(im)
    L = len(seg)
    ylo, yhi = yr
    xden = max(xhi, 1)
    xs = np.arange(L) / xden * (w - 1)
    ys = (yhi - np.asarray(seg, float)) / ((yhi - ylo) + 1e-12) * (h - 1)
    lw = max(1, int(round(1.5 * 100 / 72 * supersample)))
    d.line(list(zip(xs.tolist(), ys.tolist())), fill=0, width=lw, joint="curve")
    return im.resize((W, H), Image.LANCZOS).convert("RGB")


def render_beat(seg, lo, hi, vscale):
    """Render a beat on the train-only range [lo, hi] scaled by ``vscale``."""
    c = (lo + hi) / 2.0
    half = (hi - lo) / 2.0 or 1.0
    yr = (c - vscale * half, c + vscale * half)
    return draw_segment(seg, len(seg) - 1, yr)


def preprocess(images, device):
    """CLIPProcessor bypass: 224x224 PIL images -> normalized NCHW tensor."""
    arr = np.stack([np.asarray(im, np.float32) for im in images]) / 255.0
    arr = (arr - _CLIP_MEAN) / _CLIP_STD
    return torch.from_numpy(arr.transpose(0, 3, 1, 2)).to(device)
