import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .vision import classification_metrics


METHODS = [
    "topofault",
    "waveform_only",
    "graph_only",
    "late_fusion",
    "no_consistency",
    "no_severity",
    "no_calibration",
]


class Model(nn.Module):
    def __init__(self, cfg, method):
        super().__init__()
        self.method = method
        h = cfg["hidden"]
        self.node = nn.Linear(cfg["node_features"], h)
        self.eps = nn.Parameter(torch.zeros(2))
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, h)) for _ in range(2)]
        )
        self.wave = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(4, h, k, padding=k // 2),
                    nn.GELU(),
                    nn.Conv1d(h, h, k, padding=k // 2),
                    nn.GELU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                for k in (3, 5, 9)
            ]
        )
        self.q = nn.Linear(3 * h, h)
        self.k = nn.Linear(h, h)
        self.head = nn.Sequential(nn.Linear(5 * h, h), nn.GELU(), nn.Linear(h, 1))
        self.severity = nn.Sequential(nn.Linear(5 * h, h), nn.GELU(), nn.Linear(h, 1))
        self.wave_head = nn.Linear(3 * h, cfg["components"])
        self.confidence = nn.Sequential(nn.Linear(5 * h, 1), nn.Sigmoid())

    def forward(self, waveform, node, adjacency):
        h = self.node(node.float())
        for i, layer in enumerate(self.layers):
            h = layer((1 + self.eps[i]) * h + adjacency.float() @ h)
        wave = torch.cat([branch(waveform.float()).squeeze(-1) for branch in self.wave], -1)
        if self.method == "graph_only":
            wave = torch.zeros_like(wave)
        attention = torch.softmax(
            torch.einsum("bd,bnd->bn", self.q(wave), self.k(h)) / h.shape[-1] ** 0.5, -1
        )
        weighted = h * attention[..., None]
        if self.method == "late_fusion":
            weighted = torch.zeros_like(weighted)
        features = torch.cat([h, wave[:, None].expand(-1, h.shape[1], -1), weighted], -1)
        logits = (
            self.wave_head(wave)
            if self.method == "waveform_only"
            else self.head(features).squeeze(-1)
        )
        severity = self.severity(features).squeeze(-1)
        confidence = self.confidence(features).squeeze(-1)
        return logits, severity, confidence


def loss(model, batch, cfg):
    logits, severity, confidence = model(batch["x"], batch["node"], batch["adjacency"])
    label = batch["y"].long()
    result = F.cross_entropy(logits, label)
    if model.method != "no_severity":
        result = result + cfg["severity_weight"] * F.huber_loss(
            severity.gather(1, label[:, None]).squeeze(1), batch["severity"].float()
        )
    if model.method != "no_calibration":
        target = F.one_hot(label, logits.shape[-1]).float()
        result = result + cfg["calibration_weight"] * F.mse_loss(confidence, target)
    if model.method != "no_consistency":
        paired, _, _ = model(batch["x_pair"], batch["node"], batch["adjacency"])
        p, q = logits.softmax(-1), paired.softmax(-1)
        mean = (p + q) / 2
        divergence = (
            0.5
            * (
                (p * (p.clamp_min(1e-8).log() - mean.clamp_min(1e-8).log())).sum(-1)
                + (q * (q.clamp_min(1e-8).log() - mean.clamp_min(1e-8).log())).sum(-1)
            ).mean()
        )
        result = result + cfg["consistency_weight"] * divergence
    return result


def predict(model, batch, cfg, calibration=None):
    logits, severity, confidence = model(batch["x"], batch["node"], batch["adjacency"])
    return {"logits": logits, "severity": severity, "confidence": confidence}


def metrics(prediction, data, cfg):
    report = classification_metrics(prediction["logits"], data["y"])
    chosen = prediction["logits"].argmax(-1)
    report["severity_mae"] = float(
        np.mean(np.abs(prediction["severity"][np.arange(len(chosen)), chosen] - data["severity"]))
    )
    confidence = prediction["confidence"][np.arange(len(chosen)), chosen]
    report["confidence_brier"] = float(np.mean((confidence - (chosen == data["y"])) ** 2))
    return report


def inference(model, batch, cfg):
    return model(batch["x"], batch["node"], batch["adjacency"])
