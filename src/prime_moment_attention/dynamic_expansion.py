#!/usr/bin/env python3
"""
PRIME Dynamic Parameter Expansion Engine
========================================
Implements Function-Preserving Zero-Residual Width Expansion (Net2Net)
and Progressive Synaptic Allocation for PRIME Language Models.

Enables on-the-fly model expansion (e.g. 125M -> 180M -> 250M) without
losing prior knowledge, maintaining exact numerical equivalence (0.000000 drift)
at the moment of expansion, while initializing new plastic capacity with zero
Fisher Information curvature for incoming high-volume information injections.
"""

import math
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model: nn.Module) -> int:
    """Counts total trainable and non-trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def expand_model_mlp_width(
    model: nn.Module,
    learner: Optional[Any] = None,
    new_intermediate_size: int = 3072,
    device: Optional[torch.device] = None,
    include_in_learner: bool = True
) -> Dict[str, Any]:
    """
    Expands the SwiGLU MLP intermediate dimension of all transformer blocks
    using Net2Net function-preserving transformations.

    Mathematical Parity Guarantee:
    Since the newly appended columns in down_proj are strictly initialized to 0.0,
    for all inputs x:
        Output(x)_expanded == Output(x)_original (exact to float32 precision).
    """
    # Detect device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    old_total_params = count_parameters(model)

    # Locate layers list (handles both standalone and HF wrapped models)
    if hasattr(model, "layers"):
        layers = model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise AttributeError("Could not locate transformer blocks ('layers') in model.")

    expanded_layers_count = 0
    added_params_count = 0

    for idx, layer in enumerate(layers):
        if not hasattr(layer, "mlp"):
            continue

        mlp = layer.mlp
        gate = mlp.gate_proj
        up = mlp.up_proj
        down = mlp.down_proj

        d_in = gate.in_features
        d_old = gate.out_features
        d_new = new_intermediate_size

        if d_new <= d_old:
            raise ValueError(
                f"Target intermediate size ({d_new}) must be strictly greater than current size ({d_old})."
            )

        delta_d = d_new - d_old

        with torch.no_grad():
            orig_dtype = gate.weight.dtype

            # 1. Expand Gate Projection (d_in -> d_new)
            # Old rows are preserved; new rows initialized with small Gaussian noise
            new_gate_w = torch.empty((d_new, d_in), dtype=orig_dtype, device=device)
            new_gate_w[:d_old, :] = gate.weight.data
            nn.init.normal_(new_gate_w[d_old:, :], mean=0.0, std=0.02)

            # 2. Expand Up Projection (d_in -> d_new)
            new_up_w = torch.empty((d_new, d_in), dtype=orig_dtype, device=device)
            new_up_w[:d_old, :] = up.weight.data
            nn.init.normal_(new_up_w[d_old:, :], mean=0.0, std=0.02)

            # 3. Expand Down Projection (d_new -> d_in)
            # Old columns are preserved; NEW COLUMNS ARE STRICTLY ZERO!
            new_down_w = torch.empty((d_in, d_new), dtype=orig_dtype, device=device)
            new_down_w[:, :d_old] = down.weight.data
            new_down_w[:, d_old:].zero_()  # Critical: zero output contribution at initialization

            # Create new nn.Linear modules
            new_gate = nn.Linear(d_in, d_new, bias=False).to(device=device, dtype=orig_dtype)
            new_up = nn.Linear(d_in, d_new, bias=False).to(device=device, dtype=orig_dtype)
            new_down = nn.Linear(d_new, d_in, bias=False).to(device=device, dtype=orig_dtype)

            new_gate.weight.copy_(new_gate_w)
            new_up.weight.copy_(new_up_w)
            new_down.weight.copy_(new_down_w)

            # Assign back to MLP block
            mlp.gate_proj = new_gate
            mlp.up_proj = new_up
            mlp.down_proj = new_down

            expanded_layers_count += 1
            added_params_count += (delta_d * d_in * 3)

    # Update model configuration
    if hasattr(model, "config"):
        model.config.intermediate_size = new_intermediate_size
    if hasattr(model, "model") and hasattr(model.model, "config"):
        model.model.config.intermediate_size = new_intermediate_size

    new_total_params = count_parameters(model)

    # Synchronize with Online TTT Learner if attached
    if learner is not None:
        synchronize_learner_after_expansion(learner, model, include_in_learner=include_in_learner)

    return {
        "status": "EXPANSION_SUCCESS",
        "old_total_parameters": old_total_params,
        "new_total_parameters": new_total_params,
        "parameters_added": new_total_params - old_total_params,
        "new_intermediate_size": new_intermediate_size,
        "expanded_layers": expanded_layers_count,
        "growth_percent": round(((new_total_params - old_total_params) / old_total_params) * 100.0, 2)
    }


def synchronize_learner_after_expansion(learner: Any, model: nn.Module, include_in_learner: bool = True):
    """
    Synchronizes OnlineTTTContinualLearner state after parameter expansion:
    1. Re-registers parameter pointers in adapted_params.
    2. Inherits prior Fisher curvature for old slices.
    3. Initializes brand-new parameter slices with ZERO Fisher curvature (100% plastic).
    """
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if parameter is currently tracked or should be added
        if name in learner.adapted_params:
            old_param = learner.adapted_params[name]
            old_anchor = learner.anchor_weights[name]
            old_fisher = learner.fisher_diag[name]

            if old_param.shape != param.shape:
                # Shape expanded!
                new_anchor = param.detach().clone()
                new_fisher = torch.zeros_like(param)

                # Copy over overlapping slices
                if param.dim() == 2 and old_param.dim() == 2:
                    min_r = min(old_param.shape[0], param.shape[0])
                    min_c = min(old_param.shape[1], param.shape[1])
                    new_anchor[:min_r, :min_c] = old_anchor[:min_r, :min_c]
                    new_fisher[:min_r, :min_c] = old_fisher[:min_r, :min_c]

                learner.adapted_params[name] = param
                learner.anchor_weights[name] = new_anchor
                learner.fisher_diag[name] = new_fisher
            else:
                learner.adapted_params[name] = param
        elif include_in_learner and any(k in name for k in ["gate_proj", "up_proj", "down_proj"]):
            # Include newly expanded MLP weights into active plastic adaptation pool
            learner.adapted_params[name] = param
            learner.anchor_weights[name] = param.detach().clone()
            learner.fisher_diag[name] = torch.zeros_like(param)  # Fresh plasticity!
