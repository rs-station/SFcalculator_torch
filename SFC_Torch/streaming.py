"""Memory-bounded protein structure factors, computed in tiles of reflections.

The dense ``F_protein`` builds several [N_atoms, N_HKLs] tensors at once. Here the
reflections are processed in tiles of ``chunk_size``, so the working memory scales
with N_atoms * chunk_size instead. When gradients are needed, each tile is
checkpointed: backward recomputes the tile instead of keeping its intermediates,
trading extra computation for memory.
"""

from numbers import Integral

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


def check_chunk_size(chunk_size, name="chunk_size"):
    """Return ``chunk_size`` as an int; reject booleans, non-integers and values <= 0."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral):
        raise TypeError(f"{name} must be a positive integer")
    if chunk_size <= 0:
        raise ValueError(f"{name} must be positive")
    return int(chunk_size)


def _streamed_tile(
    hkl_tile,
    dr2_tile,
    start,
    stop,
    scattering_by_type,
    atom_type_indices,
    rotations,
    translations,
    orth2frac,
    positions_frac,
    b_iso,
    aniso_u,
    occupancy,
):
    """Dense ``F_protein`` for reflections ``start:stop``.

    The per-atom scattering table is gathered here, so it is only ever built for one tile.
    """
    # Imported here because Fmodel imports this module.
    from .Fmodel import F_protein

    atomic_scattering = scattering_by_type[:, start:stop].index_select(0, atom_type_indices)
    return F_protein(
        hkl_tile,
        dr2_tile,
        atomic_scattering,
        rotations,
        translations,
        orth2frac,
        positions_frac,
        b_iso,
        aniso_u,
        occupancy,
    )


def f_protein_streamed(
    hkl: np.ndarray,
    dr2: np.ndarray,
    scattering_by_type: torch.Tensor,
    atom_type_indices: torch.Tensor,
    rotations: torch.Tensor,
    translations: torch.Tensor,
    orth2frac: torch.Tensor,
    positions_frac: torch.Tensor,
    b_iso: torch.Tensor,
    aniso_u: torch.Tensor,
    occupancy: torch.Tensor,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """
    Memory-bounded equivalent of `SFC_Torch.Fmodel.F_protein`, computed in tiles of reflections.

    Parameters
    ----------
    hkl: np.ndarray, [N_HKLs, 3]
        Miller indices

    dr2: np.ndarray, [N_HKLs,]
        1/d^2 of each reflection

    scattering_by_type: tensor, [N_types, N_HKLs]
        Atomic scattering factors per atom type (rather than per atom), float32 or
        complex64 with anomalous scattering

    atom_type_indices: int64 tensor, [N_atoms,]
        Row of ``scattering_by_type`` belonging to each atom

    rotations, translations, orth2frac, positions_frac, b_iso, aniso_u, occupancy:
        float32 tensors, as for ``F_protein``; all tensors share one device

    chunk_size: int, default 2048
        Number of reflections per tile

    Returns
    -------
    complex64 tensor, [N_HKLs,], in the order of ``hkl``

    Each tile runs the unchanged ``F_protein``, so the result matches the dense
    calculation, and higher-order gradients are supported. Backward recomputes the
    tiles from the inputs, so do not modify them (including the NumPy arrays) before
    backward has finished.
    """
    chunk_size = check_chunk_size(chunk_size)

    inputs = (
        scattering_by_type,
        atom_type_indices,
        rotations,
        translations,
        orth2frac,
        positions_frac,
        b_iso,
        aniso_u,
        occupancy,
    )
    needs_checkpoint = torch.is_grad_enabled() and any(t.requires_grad for t in inputs)
    n_reflections = len(hkl)
    tiles = []
    # Without reflections, still run one empty tile so the result has the usual
    # dtype/device and stays connected to the inputs' autograd graph.
    for start in range(0, max(n_reflections, 1), chunk_size):
        stop = min(start + chunk_size, n_reflections)
        tile_args = (hkl[start:stop], dr2[start:stop], start, stop, *inputs)
        if needs_checkpoint:
            tile = checkpoint(
                _streamed_tile,
                *tile_args,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            tile = _streamed_tile(*tile_args)
        tiles.append(tile)

    return torch.cat(tiles, dim=0)
