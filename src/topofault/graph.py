"""Convert a flat SPICE netlist and measured component descriptors into a graph."""

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np


def build(netlist, metadata):
    """Return ordered candidate features and shared-terminal adjacency."""
    lines = []
    for raw in Path(netlist).read_text(encoding="utf-8").splitlines():
        text = raw.split("$", 1)[0].strip()
        if not text or text.startswith(("*", ".")):
            continue
        if text.startswith("+"):
            if not lines:
                raise ValueError("SPICE continuation has no preceding component")
            lines[-1] += " " + text[1:]
        else:
            lines.append(text)
    records = {tokens[0].upper(): tokens for tokens in map(str.split, lines)}
    components = metadata["components"]
    names = [component["name"].upper() for component in components]
    if len(set(names)) != len(names):
        raise ValueError("Candidate component names must be unique")
    types = {"R": 0, "C": 1, "L": 2, "D": 3, "Q": 4, "M": 5, "X": 6, "V": 7, "I": 8}
    default_pins = {"R": 2, "C": 2, "L": 2, "D": 2, "Q": 3, "M": 4, "V": 2, "I": 2}
    terminals = []
    for component, name in zip(components, names):
        if name not in records or name[0] not in types:
            raise ValueError("Missing/unsupported flat SPICE component: " + name)
        count = component.get("pin_count", default_pins.get(name[0]))
        if count is None:
            raise ValueError("Declare pin_count for subcircuit instance " + name)
        terminals.append(set(token.lower() for token in records[name][1:1 + count]))
        if len(records[name]) <= count:
            raise ValueError("Incomplete component terminal list: " + name)
    adjacency = np.array([[int(i != j and bool(left & right)) for j, right in enumerate(terminals)]
                          for i, left in enumerate(terminals)], dtype=np.float32)

    def distances(port):
        distance = np.full(len(names), len(names) + 1, dtype=float)
        queue = deque(index for index, nets in enumerate(terminals) if port.lower() in nets)
        if not queue:
            raise ValueError("Port net does not touch any candidate: " + port)
        for index in queue:
            distance[index] = 0
        while queue:
            index = queue.popleft()
            for other in np.flatnonzero(adjacency[index]):
                if distance[other] > distance[index] + 1:
                    distance[other] = distance[index] + 1
                    queue.append(other)
        return distance

    input_distance, output_distance = distances(metadata["input_net"]), distances(metadata["output_net"])
    node = np.array([[types[name[0]], np.log(max(abs(component["nominal"]), 1e-12)),
                      component["tolerance"], component["dc_voltage"], component["dc_current"],
                      component["sensitivity"], input_distance[index], output_distance[index]]
                     for index, (component, name) in enumerate(zip(components, names))], dtype=np.float32)
    if not np.isfinite(node).all():
        raise ValueError("Component descriptors must be finite")
    return node, adjacency, np.asarray(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--netlist", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    node, adjacency, names = build(args.netlist, json.loads(Path(args.metadata).read_text(encoding="utf-8")))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(str(output))
    np.savez_compressed(output, node=node, adjacency=adjacency, components=names)


if __name__ == "__main__":
    main()
