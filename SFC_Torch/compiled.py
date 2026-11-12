"""CUDA variant of the streamed backend, with each tile compiled by ``torch.compile``.

Tiling and checkpointing work as in :mod:`SFC_Torch.streaming`, but the per-tile
arithmetic uses real tensors only, so Inductor can compile it as one graph. Compiling
needs the CUDA toolkit (``ptxas``), takes time on first use and supports first-order
gradients only; use the streamed backend for higher-order derivatives. Tested with
PyTorch 2.7.
"""

from functools import lru_cache
from math import pi
from types import FunctionType

import torch
from torch.utils.checkpoint import checkpoint

from .streaming import check_chunk_size

__all__ = ["f_protein_compiled"]


def _tensor_tile(
    hkl,
    dr2,
    scattering_real,
    scattering_imag,
    atom_type_indices,
    rotations,
    translations,
    orth2frac,
    positions_frac,
    b_iso,
    aniso_u,
    occupancy,
    isotropic_only=False,
):
    """Structure factors of one tile as ``(real, imag)``, summed over atoms.

    The inputs are already sliced to the tile; ``scattering_imag`` is None without
    anomalous scattering. The arithmetic follows ``F_protein``, but without complex
    tensors so the function compiles as one graph; the loop over symmetry operations
    is unrolled during compilation.

    ``isotropic_only=True`` skips the anisotropic Debye-Waller factor. Only pass it
    when ``aniso_u`` is all zero and neither ``aniso_u`` nor ``orth2frac`` requires
    grad: otherwise their gradients must still be computed (as zeros).
    """
    sf_real = scattering_real.index_select(0, atom_type_indices) * occupancy[:, None]
    sf_imag = None
    if scattering_imag is not None:
        sf_imag = scattering_imag.index_select(0, atom_type_indices) * occupancy[:, None]
    dwf_iso = torch.exp(-b_iso[:, None] * dr2 / 4.0)
    sym_positions = torch.einsum("oxy,ay->aox", rotations, positions_frac) + translations
    if not isotropic_only:
        mask = torch.all(torch.all(aniso_u == 0.0, dim=-1), dim=-1)
        # Same contraction and operand order as `DWF_aniso`, to match its rounding.
        ustar = torch.einsum("xy,ayz,wz->axw", orth2frac, aniso_u, orth2frac)
        sym_hkl = torch.einsum("rx,oxy->roy", hkl, rotations)
    phase_real = torch.zeros_like(sf_real)
    phase_imag = torch.zeros_like(sf_real)
    for i in range(rotations.shape[0]):
        xyz = sym_positions[:, i, :]
        phase = (2 * pi) * (
            xyz[:, 0, None] * hkl[None, :, 0]
            + xyz[:, 1, None] * hkl[None, :, 1]
            + xyz[:, 2, None] * hkl[None, :, 2]
        )
        damping = dwf_iso
        if not isotropic_only:
            # h^T U* h for every atom and reflection.
            h, k, ell = (sym_hkl[None, :, i, j] for j in range(3))
            quadratic = (
                (h * ustar[:, 0, 0, None] + k * ustar[:, 1, 0, None] + ell * ustar[:, 2, 0, None])
                * h
                + (h * ustar[:, 0, 1, None] + k * ustar[:, 1, 1, None] + ell * ustar[:, 2, 1, None])
                * k
                + (h * ustar[:, 0, 2, None] + k * ustar[:, 1, 2, None] + ell * ustar[:, 2, 2, None])
                * ell
            )
            dwf_aniso = torch.exp((-2.0 * pi**2) * quadratic)
            damping = torch.where(mask[:, None], dwf_iso, dwf_aniso)
        phase_real = phase_real + damping * torch.cos(phase)
        phase_imag = phase_imag + damping * torch.sin(phase)
    if sf_imag is None:
        return (phase_real * sf_real).sum(dim=0), (phase_imag * sf_real).sum(dim=0)
    return (
        (phase_real * sf_real - phase_imag * sf_imag).sum(dim=0),
        (phase_imag * sf_real + phase_real * sf_imag).sum(dim=0),
    )


def _tile_signature(args):
    """Hashable key of what compilation specializes on.

    Grad and inference mode, plus each tensor argument's shape, stride, dtype, device
    and ``requires_grad``; other arguments by value. Tensor values are not included.
    """
    return (
        torch.is_grad_enabled(),
        torch.is_inference_mode_enabled(),
        tuple(
            (arg.shape, arg.stride(), arg.dtype, arg.device, arg.requires_grad)
            if isinstance(arg, torch.Tensor)
            else arg
            for arg in args
        ),
    )


@lru_cache(maxsize=32)
def _compiled_tile(signature):
    """Return a compiled copy of ``_tensor_tile`` for one ``_tile_signature``.

    ``torch.compile`` keeps its compiled variants per Python code object, and stops
    recompiling a code object once Dynamo's recompile limit is reached. Different
    models, the shorter last tile and grad/no-grad each need their own variant, so
    each signature gets its own copy of the code object instead of raising the global
    limit. At most 32 copies are kept. Use ``TORCH_LOGS=recompiles`` to inspect
    unexpected recompilations (e.g. caused by autocast).
    """
    tile = FunctionType(
        _tensor_tile.__code__.replace(),
        _tensor_tile.__globals__,
        _tensor_tile.__name__,
        _tensor_tile.__defaults__,
    )
    return torch.compile(tile, fullgraph=True, dynamic=False, backend="inductor")


def _check_compiled_inputs(
    hkl_tensor, dr2_tensor, scattering_by_type, atom_type_indices, parameters, compile_kernel
):
    """Validate devices, dtypes and shapes up front, rather than failing inside compiled code."""
    if not isinstance(compile_kernel, bool):
        raise TypeError("compile_kernel must be a bool")
    if compile_kernel and hkl_tensor.device.type != "cuda":
        raise ValueError("compile_kernel=True requires CUDA; use compile_kernel=False on CPU")
    if hkl_tensor.dtype != torch.float32 or dr2_tensor.dtype != torch.float32:
        raise ValueError("hkl_tensor and dr2_tensor must be float32")
    n_reflections = hkl_tensor.shape[0]
    if hkl_tensor.shape != (n_reflections, 3) or dr2_tensor.shape != (n_reflections,):
        raise ValueError("hkl_tensor and dr2_tensor must have shapes (R, 3) and (R,)")
    if scattering_by_type.ndim != 2 or scattering_by_type.shape[1] != n_reflections:
        raise ValueError("scattering_by_type must have shape (n_types, R)")
    inputs = (hkl_tensor, dr2_tensor, scattering_by_type, atom_type_indices, *parameters)
    if any(t.device != hkl_tensor.device for t in inputs):
        raise ValueError("all inputs must be on the same device")
    if any(t.dtype != torch.float32 for t in parameters):
        raise ValueError("geometry and atomic parameter tensors must be float32")
    if scattering_by_type.dtype not in (torch.float32, torch.complex64):
        raise ValueError("scattering_by_type must be float32 or complex64")
    if atom_type_indices.dtype != torch.int64:
        raise ValueError("atom_type_indices must be int64")


def f_protein_compiled(
    hkl_tensor: torch.Tensor,
    dr2_tensor: torch.Tensor,
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
    *,
    compile_kernel: bool = True,
) -> torch.Tensor:
    """
    Variant of `SFC_Torch.streaming.f_protein_streamed` with compiled tiles.

    Parameters are as for ``f_protein_streamed``, except:

    hkl_tensor, dr2_tensor: float32 tensors, [N_HKLs, 3] and [N_HKLs,]
        Already on the device, so they are not copied on every call

    compile_kernel: bool, default True
        Compile the tile arithmetic (CUDA only). False runs the same arithmetic
        eagerly on any device, for testing. A compilation failure is raised, never
        silently replaced by eager execution.

    Returns
    -------
    complex64 tensor, [N_HKLs,]

    All tensors share one device. Only first-order gradients are supported. Backward
    recomputes the tiles from the inputs, so do not modify them before it has finished.

    If ``aniso_u`` is all zero and neither it nor ``orth2frac`` requires grad, the
    anisotropic term is skipped (the check synchronizes with the GPU once per call).
    Otherwise the full masked computation runs, so gradients with respect to an
    all-zero trainable ``aniso_u`` are zero tensors rather than missing.
    """
    chunk_size = check_chunk_size(chunk_size)
    parameters = (
        rotations,
        translations,
        orth2frac,
        positions_frac,
        b_iso,
        aniso_u,
        occupancy,
    )
    _check_compiled_inputs(
        hkl_tensor, dr2_tensor, scattering_by_type, atom_type_indices, parameters, compile_kernel
    )
    inputs = (hkl_tensor, dr2_tensor, scattering_by_type, atom_type_indices, *parameters)
    needs_checkpoint = torch.is_grad_enabled() and any(t.requires_grad for t in inputs)
    isotropic_only = (
        not aniso_u.requires_grad
        and not orth2frac.requires_grad
        and bool(torch.all(aniso_u == 0.0))
    )

    if scattering_by_type.is_complex():
        scattering_real, scattering_imag = scattering_by_type.real, scattering_by_type.imag
    else:
        scattering_real, scattering_imag = scattering_by_type, None
    n_reflections = hkl_tensor.shape[0]
    tiles = []
    # Without reflections, still run one empty tile so the result has the usual
    # dtype/device and stays connected to the inputs' autograd graph.
    for start in range(0, max(n_reflections, 1), chunk_size):
        stop = min(start + chunk_size, n_reflections)
        args = (
            hkl_tensor[start:stop],
            dr2_tensor[start:stop],
            scattering_real[:, start:stop],
            None if scattering_imag is None else scattering_imag[:, start:stop],
            atom_type_indices,
            *parameters,
            isotropic_only,
        )
        evaluate = _compiled_tile(_tile_signature(args)) if compile_kernel else _tensor_tile
        if needs_checkpoint:
            real, imag = checkpoint(evaluate, *args, use_reentrant=False, preserve_rng_state=False)
        else:
            real, imag = evaluate(*args)
        tiles.append(torch.complex(real, imag))
    return torch.cat(tiles, dim=0)
