import numpy as np

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def expected_calibration_error(confidence, correct, bins=15):
    """Equal-width ECE with each finite confidence assigned to exactly one bin."""
    confidence = np.asarray(confidence)
    correct = np.asarray(correct)
    indices = np.clip(np.searchsorted(np.linspace(0, 1, bins + 1), confidence, side="left") - 1,
                      0, bins - 1)
    return float(sum(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
                     for index in range(bins) if (mask := indices == index).any()))


def classification_metrics(logits, labels):
    logits = np.asarray(logits)
    probability = np.exp(logits - logits.max(-1, keepdims=True))
    probability /= probability.sum(-1, keepdims=True)
    labels = np.asarray(labels).astype(int)
    prediction = probability.argmax(-1)
    confidence = probability.max(-1)
    correct = prediction == labels
    ece = expected_calibration_error(confidence, correct)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, labels=np.arange(probability.shape[-1]), average="macro", zero_division=0)),
        "top3": float(
            np.mean([y in row for y, row in zip(labels, np.argsort(-probability, axis=-1)[:, :3])])
        ),
        "ece": float(ece),
        "nll": float(
            -np.log(np.maximum(probability[np.arange(len(labels)), labels], 1e-12)).mean()
        ),
    }
