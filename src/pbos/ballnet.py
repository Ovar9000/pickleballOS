"""Learned ball detector: a small U-Net over three stacked frames (TrackNet idea, our own code).

Input: frames t-2, t-1, t (BGR, /255) stacked into 9 channels at full resolution. Only past frames,
so it stays causal for live use. Output: a heatmap for frame t; peaks are ball candidates that feed
the same BallTracker as the classical detector. The tracker still adds physics (arcs, coasting).

Full resolution matters: at 720p the ball is 3-8 px, and halving the resolution (as TrackNet does
for broadcast tennis) would leave it 1-4 px. Training uses 256 px crops; the network is fully
convolutional, so inference runs on the whole court ROI.

Training needs PyTorch (BSD-3). Inference uses the exported ONNX through ONNX Runtime.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .balldata import BallLabel


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class BallNet(nn.Module):
    """U-Net, 3 down / 3 up levels. Input H and W must be multiples of 8."""

    def __init__(self, c=(16, 32, 64, 128)):
        super().__init__()
        self.e1, self.e2, self.e3, self.e4 = _block(9, c[0]), _block(c[0], c[1]), _block(c[1], c[2]), _block(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.u3, self.d3 = nn.ConvTranspose2d(c[3], c[2], 2, 2), _block(2 * c[2], c[2])
        self.u2, self.d2 = nn.ConvTranspose2d(c[2], c[1], 2, 2), _block(2 * c[1], c[1])
        self.u1, self.d1 = nn.ConvTranspose2d(c[1], c[0], 2, 2), _block(2 * c[0], c[0])
        self.head = nn.Conv2d(c[0], 1, 1)
        nn.init.constant_(self.head.bias, -4.6)  # start near p = 0.01 everywhere, standard for focal loss

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        x4 = self.e4(self.pool(x3))
        y = self.d3(torch.cat([self.u3(x4), x3], 1))
        y = self.d2(torch.cat([self.u2(y), x2], 1))
        y = self.d1(torch.cat([self.u1(y), x1], 1))
        return self.head(y)


def focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CenterNet-style focal loss on a Gaussian target heatmap (peak = 1 at the ball centre)."""
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = target.ge(0.999).float()
    pos_loss = torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = torch.log(1 - p) * p ** 2 * (1 - target) ** 4 * (1 - pos)
    return -(pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1.0)


def gaussian_heatmap(h: int, w: int, u: float, v: float, sigma: float) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.exp(-((xs - u) ** 2 + (ys - v) ** 2) / (2 * sigma**2))
    iu, iv = int(round(u)), int(round(v))
    if 0 <= iu < w and 0 <= iv < h:
        g[iv, iu] = 1.0  # exact peak, so the focal loss has a positive pixel
    return g


class TripletCrops(Dataset):
    def __init__(self, labels: list[BallLabel], frame_dir: str | Path, crop: int = 256, train: bool = True,
                 sigma: float = 2.0, image_size: tuple[int, int] = (1280, 720)):
        self.labels = [lb for lb in labels if lb.visible]
        self.dir = Path(frame_dir)
        self.crop, self.train, self.sigma = crop, train, sigma
        self.w, self.h = image_size

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        lb = self.labels[i]
        c = self.crop
        rng = random if self.train else random.Random(i)
        # Ball anywhere in the crop except the outer 16 px (train), centred-ish (val, deterministic).
        x0 = int(np.clip(lb.u - rng.uniform(16, c - 16), 0, self.w - c))
        y0 = int(np.clip(lb.v - rng.uniform(16, c - 16), 0, self.h - c))
        frames = [cv2.imread(str(self.dir / f"{lb.frame - k:06d}.jpg"))[y0:y0 + c, x0:x0 + c] for k in (2, 1, 0)]
        x = np.concatenate(frames, axis=2).astype(np.float32) / 255.0
        u, v = lb.u - x0, lb.v - y0
        if self.train:
            x = np.clip(x * rng.uniform(0.75, 1.25) + rng.uniform(-0.08, 0.08), 0, 1)  # same jitter for all 3
            if rng.random() < 0.5:
                x = x[:, ::-1]
                u = c - 1 - u
        y = gaussian_heatmap(c, c, u, v, self.sigma)
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), torch.from_numpy(y[None])


def peak_metrics(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5, tol_px: float = 4.0) -> tuple[int, int]:
    """(hits, n): a hit is a peak above thr within tol_px of the true centre."""
    p = torch.sigmoid(logits.float())[:, 0]
    t = target[:, 0]
    hits = 0
    for k in range(p.shape[0]):
        pv, pi = p[k].flatten().max(0)
        ti = t[k].flatten().argmax()
        w = p.shape[-1]
        d = math.hypot(int(pi) % w - int(ti) % w, int(pi) // w - int(ti) // w)
        hits += int(pv >= thr and d <= tol_px)
    return hits, p.shape[0]


def train(train_labels: list[BallLabel], val_labels: list[BallLabel], frame_dir: str | Path, out_dir: str | Path,
          epochs: int = 12, batch: int = 32, lr: float = 2e-3, workers: int = 4, crop: int = 256,
          log=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    random.seed(0)
    model = BallNet().to(dev)
    tl = DataLoader(TripletCrops(train_labels, frame_dir, crop=crop, train=True), batch_size=batch, shuffle=True,
                    num_workers=workers, persistent_workers=workers > 0, pin_memory=True, drop_last=True)
    vl = DataLoader(TripletCrops(val_labels, frame_dir, crop=crop, train=False), batch_size=batch, shuffle=False,
                    num_workers=workers, persistent_workers=workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(tl))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    best, history = -1.0, []
    for ep in range(epochs):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                logits = model(x)
            loss = focal_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += float(loss)
        model.eval()
        hits = n = 0
        with torch.no_grad():
            for x, y in vl:
                with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    logits = model(x.to(dev))
                h, k = peak_metrics(logits, y.to(dev))
                hits, n = hits + h, n + k
        acc = hits / max(n, 1)
        history.append({"epoch": ep + 1, "loss": round(total / len(tl), 4), "val_hit_rate": round(acc, 4),
                        "seconds": round(time.time() - t0, 1)})
        log(history[-1])
        if acc > best:
            best = acc
            torch.save(model.state_dict(), out_dir / "ballnet.pt")
    model.load_state_dict(torch.load(out_dir / "ballnet.pt", map_location=dev))
    export_error = None
    try:
        export_onnx(model.cpu().eval(), out_dir / "ballnet.onnx")
    except Exception as e:  # e.g. the `onnx` package missing in this environment; the checkpoint is safe
        export_error = f"{type(e).__name__}: {e}"
        log(f"ONNX export failed ({export_error}). The checkpoint is saved; export later with: "
            f"python -c \"import torch; from pbos.ballnet import BallNet, export_onnx; m = BallNet(); "
            f"m.load_state_dict(torch.load('{out_dir / 'ballnet.pt'}', map_location='cpu')); "
            f"export_onnx(m.eval(), '{out_dir / 'ballnet.onnx'}')\"")
    return {"export_error": export_error, "best_val_hit_rate": best, "history": history, "device": dev.type, "crop": crop, "epochs": epochs,
            "train_labels": len(train_labels), "val_labels": len(val_labels)}


class _Deployable(nn.Module):
    """Takes the three raw uint8 frames stacked as (1, H, W, 9) and returns probabilities.

    Doing the layout change and the /255 inside the graph means the caller only stacks bytes: the
    float conversion of ~27 MB per 720p frame triple was the slowest part of inference in numpy.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.sigmoid(self.net(x.permute(0, 3, 1, 2).float() / 255.0))


def export_onnx(model: nn.Module, path: str | Path) -> None:
    dummy = torch.zeros(1, 256, 256, 9, dtype=torch.uint8)
    kwargs = dict(input_names=["frames_u8"], output_names=["heatmap"], opset_version=17,
                  dynamic_axes={"frames_u8": {1: "h", 2: "w"}, "heatmap": {2: "h", 3: "w"}})
    try:
        torch.onnx.export(_Deployable(model), dummy, str(path), dynamo=False, **kwargs)
    except TypeError:  # older torch without the dynamo switch
        torch.onnx.export(_Deployable(model), dummy, str(path), **kwargs)
