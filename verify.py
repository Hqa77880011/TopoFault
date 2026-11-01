"""CPU functional checks on generated arrays."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))
from topofault import core
from topofault.cli import Arrays, evaluate, plot, prepare, train
from topofault.vision import classification_metrics
from topofault.graph import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    rng = np.random.default_rng(17)
    netlist = output / "three-resistors.cir"
    netlist.write_text("Functional graph\nR1 in n1 1k\nR2 n1 out 2k\nR3 out 0 3k\n.end\n", encoding="utf-8")
    metadata = dict(input_net="in", output_net="out", components=[
        dict(name=f"R{index}", nominal=index * 1000, tolerance=0.05,
             dc_voltage=0.1 * index, dc_current=0.001, sensitivity=0.01) for index in (1, 2, 3)])
    graph_node, graph_adjacency, names = build(netlist, metadata)
    np.testing.assert_array_equal(graph_adjacency, [[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    np.testing.assert_array_equal(graph_node[:, 6], [0, 1, 2])
    np.testing.assert_array_equal(names, ["R1", "R2", "R3"])
    n, v, t = 36, 3, 24
    group = np.repeat(np.arange(12), 3)
    labels = group % v
    waveform = rng.normal(size=(n, 4, t)).astype(np.float32)
    waveform[:, 0] += labels[:, None]
    node = rng.normal(size=(n, v, 8)).astype(np.float32)
    adjacency = np.broadcast_to(np.ones((v, v)) - np.eye(v), (n, v, v)).astype(np.float32)
    np.savez_compressed(output / "raw.npz", x=waveform,
                        x_pair=waveform + rng.normal(0, 0.05, waveform.shape).astype(np.float32),
                        node=node, adjacency=adjacency, y=labels,
                        severity=((group // 3 + 1) * 0.05).astype(np.float32),
                        group=group, split=np.where(group < 6, 0, np.where(group < 9, 1, 2)))
    prepare(output / "raw.npz", output / "data", 17)
    cfg = json.loads((Path(__file__).parent / "configs/default.json").read_text(encoding="utf-8"))
    cfg.update(hidden=8, components=v, epochs=1, batch_size=6, learning_rate=0.01)
    batch = next(iter(torch.utils.data.DataLoader(Arrays(output / "data/train.npz"), batch_size=6)))
    model = core.Model(cfg, "topofault")
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    value = core.loss(model, batch, cfg)
    assert torch.isfinite(value)
    value.backward()
    critical = ["eps", "node.weight", "q.weight", "k.weight", "wave.0.0.weight",
                "head.0.weight", "severity.0.weight", "confidence.0.weight"]
    parameters = dict(model.named_parameters())
    for name in critical:
        gradient = parameters[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0, name
    optimizer.step()
    for name in critical:
        assert not torch.equal(before[name], parameters[name]), name
    model.eval()
    with torch.no_grad():
        expected = model(batch["x"], batch["node"], batch["adjacency"])
    torch.save(model.state_dict(), output / "roundtrip.pt")
    restored = core.Model(cfg, "topofault").eval()
    restored.load_state_dict(torch.load(output / "roundtrip.pt", weights_only=True))
    with torch.no_grad():
        for original, loaded in zip(expected, restored(batch["x"], batch["node"], batch["adjacency"])):
            torch.testing.assert_close(original, loaded, rtol=1e-6, atol=1e-7)
    # These controls must suppress graph dependence in *every* output head.
    for method in ("waveform_only",):
        control = core.Model(cfg, method).eval()
        with torch.no_grad():
            first = control(batch["x"], batch["node"], batch["adjacency"])
            second = control(batch["x"], batch["node"] * 13, batch["adjacency"] * 7)
            for left, right in zip(first, second):
                torch.testing.assert_close(left, right)
    graph = core.Model(cfg, "graph_only").eval()
    with torch.no_grad():
        for left, right in zip(graph(batch["x"], batch["node"], batch["adjacency"]),
                               graph(batch["x"] * 9, batch["node"], batch["adjacency"])):
            torch.testing.assert_close(left, right)
    hand = classification_metrics(np.log([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1],
                                          [0.1, 0.6, 0.3]]), np.array([0, 1, 2]))
    assert np.isclose(hand["accuracy"], 2 / 3)
    assert np.isclose(hand["macro_f1"], 5 / 9)
    assert np.isclose(hand["top3"], 1)
    assert np.isclose(hand["ece"], 11 / 30)
    reports = {}
    for method in core.METHODS:
        destination = output / "runs" / method
        train(cfg, output / "data", destination, method)
        reports[method] = evaluate(destination / "model.pt", output / "data/test.npz", output / "evaluation" / method)
        assert all(np.isfinite(value) for value in reports[method].values())
    plot([output / "runs/topofault", output / "evaluation/topofault"], output / "figures")
    from topofault import baselines
    classical = {}
    for method in ["svm"] + (["xgboost"] if importlib.util.find_spec("xgboost") else []):
        destination = output / "runs" / method
        baselines.train(output / "data", destination, method)
        classical[method] = baselines.evaluate(destination / "model.joblib", output / "data/test.npz",
                                              output / "evaluation" / method)
    result = {"kind": "synthetic_functional_check", "seed": 17, "initial_loss": float(value.detach()),
              "critical_parameters_updated": critical, "roundtrip_rtol": 1e-6,
              "methods": list(reports), "hand_metrics": hand, "classical_methods": classical}
    (output / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
