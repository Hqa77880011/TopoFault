# TopoFault

Implementation of **Netlist-aware graph-waveform alignment for analog soft-fault localization**.

Implements two GIN layers, parallel waveform kernels 3/5/9, response-conditioned component attention, component localization, signed severity and confidence heads with tolerance-pair JS and Brier calibration.

## Install

Use Python 3.11 or newer. Install a matching PyTorch/TorchVision build for the target CPU or CUDA device, then install the package.

```bash
git clone https://github.com/Hqa77880011/TopoFault.git
cd TopoFault
python -m venv .venv
```

Activate with `source .venv/bin/activate` or `.venv\Scripts\Activate.ps1` in PowerShell.

```bash
python -m pip install -e .
```

## Prepare data

Generate responses with ngspice or import existing SPICE Monte Carlo runs. Group every tolerance realization by circuit, injected component and signed drift before splitting.

`x` and `x_pair` are float32 [N,4,T] with AC magnitude, unwrapped AC phase, transient voltage and transient difference resampled onto a fixed normalized grid. `node` is [N,V,F], `adjacency` [N,V,V], `y` zero-based component index, `severity` signed drift and `group` fault-setting ID. Use separate configurations for circuits with different V.

```bash
python -m topofault prepare --source data/raw.npz --output data/prepared --seed 17
```

Use `group` to keep every derived view of one source object in the same partition. An explicit integer `split` array (0=train, 1=validation, 2=test) preserves a registered split. Otherwise groups are assigned 60/20/20. Every stored field must share the sample axis. Files are loaded without pickle.

## Train

Edit `configs/default.json` for the input dimensions, device and training budget.

```bash
python -m topofault train --config configs/default.json --data data/prepared --output runs/main
python -m topofault suite --config configs/default.json --data data/prepared --output runs/comparison
```

`--methods` selects a subset. topofault, waveform_only, graph_only, late_fusion, no_consistency, no_severity and no_calibration.

The common driver uses AdamW with cosine learning-rate decay, validation-objective early stopping and a fixed seed. Custom federated/continual drivers use their registered round/task budgets. It saves only a validation-selected model and measured epoch history. `learning_rate`, `weight_decay`, `batch_size`, `epochs`, `patience` and `device` control optimization. Use separate output roots for independent runs.

## Evaluate and visualize

```bash
python -m topofault evaluate --checkpoint runs/main/topofault/model.pt --data data/prepared/test.npz --output results/topofault
python -m topofault plot --inputs runs/main/topofault results/topofault --output figures
```

`model.pt` contains model parameters, configuration and method identity. `history.jsonl` contains measured training/validation loss and epoch time. Evaluation writes `predictions.npz` and `metrics.json`. Plotting creates loss curves and metric comparisons from supplied run directories.

Component Top-1/Top-3, macro F1, ECE/NLL, selected-component severity MAE, confidence Brier, parameter count and latency.

## Implementation conventions

No graph library is required. Adjacency represents components sharing a net. Include explicit component features in a fixed node order; actual measured DC/sensitivity values must be computed by the circuit frontend. The local driver operates on fixed-size circuits; padding masks are not inferred.


## Pack image records

A JSONL manifest can point to images, masks and NPY feature arrays. Paths are relative to the manifest. All rows use identical keys; group stores the source object ID. For example, an image-classification row is `{"x":"images/1.png","y":0,"group":"object-1"}`.

```bash
python -m topofault.pack --manifest data/manifest.jsonl --output data/raw.npz --channels 3 --target class
```

Inference timing runs the deployment forward pass separately from calibration, commutation and other evaluator diagnostics, after one warmup. Batch P50/P95 and examples/s retain the configured batch size. MAC/FLOP counts cover Conv/Linear/GRU multiply-adds; process RSS includes the dataset and Python runtime.
