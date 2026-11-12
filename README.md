# SFcalculator_torch
![licence](https://img.shields.io/github/license/Hekstra-Lab/SFcalculator_torch)
![Build](https://github.com/Hekstra-Lab/SFcalculator_torch/workflows/Build/badge.svg)
[![codecov](https://codecov.io/github/Hekstra-Lab/SFcalculator_torch/branch/master/graph/badge.svg?token=02GUPGPUC1)](https://codecov.io/github/Hekstra-Lab/SFcalculator_torch)
[![PyPI](https://img.shields.io/pypi/v/SFcalculator-torch?color=blue)](https://pypi.org/project/SFcalculator-torch/)
[![Tutorial](https://img.shields.io/badge/Page-Tutorial-violet)](https://rs-station.github.io/SFcalculator_torch/)
[![BioRXiv](https://img.shields.io/badge/BioRXiv-10.1101%2F2025.01.12.632630v2-b31b1b?)](https://www.biorxiv.org/content/10.1101/2025.01.12.632630v2)


A pytorch implementation of `SFcalculator`. See the [main page](https://github.com/Hekstra-Lab/SFcalculator) for more information.

To install this package:

1. Create a python environment with package manager you like ([mambaforge](https://github.com/mamba-org/mamba) recommended).

2. Install [Pytorch](https://pytorch.org/get-started/locally/)

3. Install `SFcalculator-torch`
    ```bash
    pip install SFcalculator-torch
    ```
 
For a walkthrough tutorial, check this page: https://rs-station.github.io/SFcalculator_torch/

### Memory-bounded structure factors

By default, `calc_fprotein` builds several atoms-by-reflections tensors at once, which can
run out of memory for large models. The opt-in `"streamed"` and `"compiled"` backends process
reflections in tiles instead, so this working memory scales with atoms × `reflection_chunk_size`:

```python
from SFC_Torch import SFcalculator

calculator = SFcalculator(
    "model.pdb", dmin=3.0, set_experiment=False, device="cuda:0",
    fprotein_backend="streamed", reflection_chunk_size=8192,
)
fprotein = calculator.calc_fprotein(Return=True)
```

| `fprotein_backend` | Devices | Gradients | Notes |
|---|---|---|---|
| `"dense"` (default) | CPU, CUDA | any order | Original calculation |
| `"streamed"` | CPU, CUDA | any order | Recomputes tiles during backward: less memory, more compute |
| `"compiled"` | CUDA | first order | `"streamed"` with tiles compiled by `torch.compile`; compiles on first use and needs a working CUDA `torch.compile` setup (see below) |

Larger tiles use more memory; inputs and outputs still scale with the number of reflections.
Do not modify the inputs before backward has finished. Reading or assigning `fullsf_tensor`
switches the calculator back to the dense calculation, so in-place edits to it keep working;
`calc_fprotein_batch` always uses the dense calculation.

The `"compiled"` backend needs a working CUDA `torch.compile` setup. On first use, Triton
builds a small C helper and compiles kernels, which requires a C compiler, the CUDA headers
and `ptxas`. PyTorch installed from pip usually comes with a Triton that bundles the headers
and `ptxas`. In a conda environment, install the toolkit for your CUDA version, e.g.
`cuda-nvcc` (which also brings a matching compiler) and `cuda-cudart-dev`. Compile errors
that mention `gcc`, `main.c` or `ptxas` point to this setup; you can check it with:

```sh
python -c "import torch; print(torch.compile(torch.sin)(torch.ones(1, device='cuda')))"
```

### Citing

```
@article{li2025sfcalculator,
  title={SFCalculator: connecting deep generative models and crystallography},
  author={Li, Minhuan and Dalton, Kevin M and Hekstra, Doeke Romke},
  journal={bioRxiv},
  pages={2025--01},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

