"""Train-only standardized response descriptors with SVM/XGBoost baselines."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from .vision import classification_metrics


def descriptors(waveform):
    """Twelve fixed descriptors per AC/transient channel, preserving channel order."""
    waveform = np.asarray(waveform, dtype=float)
    if waveform.ndim != 3 or waveform.shape[1] != 4 or waveform.shape[2] < 2:
        raise ValueError("Responses require [N,4,T] with T>=2")
    summary = [waveform.mean(-1), waveform.std(-1), waveform.min(-1), waveform.max(-1),
               np.median(waveform, axis=-1), np.quantile(waveform, 0.1, axis=-1),
               np.quantile(waveform, 0.9, axis=-1), waveform[..., 0], waveform[..., -1],
               (waveform[..., 1:] + waveform[..., :-1]).sum(-1) / (2 * (waveform.shape[-1] - 1)),
               np.mean(waveform**2, axis=-1), waveform.argmax(-1) / (waveform.shape[-1] - 1)]
    return np.stack(summary, axis=-1).reshape(len(waveform), -1)


def train(data, output, method, seed=17):
    """Fit descriptors and signed-drift regression on the training partition only."""
    with np.load(Path(data) / "train.npz", allow_pickle=False) as source:
        x, y, drift = descriptors(source["x"]), source["y"], source["severity"]
        components = source["node"].shape[1]
    if not np.array_equal(np.unique(y), np.arange(components)):
        raise ValueError("Every candidate class must occur in training data")
    if method == "svm":
        classifier = SVC(C=1.0, kernel="rbf", probability=True, random_state=seed)
        regressor = SVR(C=1.0, kernel="rbf", epsilon=0.01)
    else:
        from xgboost import XGBClassifier, XGBRegressor
        options = dict(n_estimators=100, max_depth=3, learning_rate=0.05,
                       n_jobs=1, random_state=seed)
        classifier = XGBClassifier(**options, objective="multi:softprob", num_class=components)
        regressor = XGBRegressor(**options, objective="reg:squarederror")
    classifier = make_pipeline(StandardScaler(), classifier).fit(x, y)
    regressor = make_pipeline(StandardScaler(), regressor).fit(x, drift)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    payload = dict(classifier=classifier, regressor=regressor, method=method, seed=seed,
                   components=components, descriptor_count=x.shape[1])
    joblib.dump(payload, output / "model.joblib")
    return payload


def evaluate(checkpoint, data, output):
    """Evaluate a trusted locally produced baseline checkpoint without refitting."""
    payload = joblib.load(checkpoint)
    with np.load(data, allow_pickle=False) as source:
        x, labels, drift = descriptors(source["x"]), source["y"], source["severity"]
    probability = payload["classifier"].predict_proba(x)
    severity = payload["regressor"].predict(x)
    metrics = classification_metrics(np.log(np.maximum(probability, 1e-12)), labels)
    metrics["severity_mae"] = float(np.mean(np.abs(severity - drift)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "predictions.npz", probability=probability, severity=severity)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    item = sub.add_parser("train")
    item.add_argument("--data", required=True)
    item.add_argument("--output", required=True)
    item.add_argument("--method", choices=["svm", "xgboost"], required=True)
    item.add_argument("--seed", type=int, default=17)
    item = sub.add_parser("evaluate")
    item.add_argument("--checkpoint", required=True)
    item.add_argument("--data", required=True)
    item.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.output, args.method, args.seed)
    else:
        print(json.dumps(evaluate(args.checkpoint, args.data, args.output), indent=2))


if __name__ == "__main__":
    main()
