"""Vendored, structure-conditioned ("truncated") diffusion sampler for Chai-1.

This module implements HalluDesign §2.1: starting from a prior structure perturbed
by sigma[ref_idx], run only the trailing ``ref_time_steps`` diffusion steps.

Vendored from chai_lab **0.6.1** (``gradio_design-gradio-design:latest`` image,
installed at ``/opt/venv`` or editable from ``/chai-lab/chai_lab/chai1.py``).
Supersedes the 0.0.1 vendor (7a3118e) which used ``ConstraintContext``,
``run_folding_on_context`` returning a 4-tuple, and ``load_exported`` with
``.pt2`` size-specific paths.

Two changes versus the original ``run_folding_on_context`` (0.6.1):

  CHANGE 1: Initial atom positions are computed from a prior structure perturbed
            by ``sigma[ref_idx]`` instead of pure noise from ``sigma[0]``.
  CHANGE 2: The Karras update loop starts at ``ref_idx`` instead of 0, running
            only the trailing diffusion steps.

New parameters (beyond the originals):
    initial_coords (torch.Tensor | None):
        (num_atoms_real, 3) tensor in Angstroms. If None, falls back to pure-noise
        init (identical to the original function).
    ref_time_steps (int):
        Number of trailing diffusion steps to run (default 50).
        ``ref_idx = max(0, num_diffn_timesteps - ref_time_steps)``.

Everything else — feature prep, trunk, confidence head, CIF+score writing — is
UNCHANGED from the 0.6.1 source.

Heavy chai_lab imports are deferred to function call time so this module loads
without chai_lab (mirroring the lazy-import pattern in chai_backend.py).

TODO (#27, constraints): this vendored truncated sampler does NOT yet support
``constraint_path`` / restraints. ``make_all_atom_feature_context`` here is called
without a ``constraints`` argument, so a restrained α run must use the full-``predict``
refine path (see scripts/design_alpha.py ``_PredictBackendWrapper``) instead of
``truncated_refine``. Threading restraints through the truncated path is non-trivial: chai
0.6.1 builds restraint features inside ``make_all_atom_feature_context`` (a
``RestraintContext`` from the parsed CSV) and the truncated init perturbs a PRIOR structure
that may already violate the contact, so the restraint feature and the structure-conditioned
init interact. Deferred deliberately rather than half-wired; ``--no_restraints`` reverts to
this (unconstrained) truncated path.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Public entry point — mirrors run_inference 0.6.1 signature + new params
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference_truncated(
    fasta_file: Path,
    *,
    output_dir: Path,
    use_esm_embeddings: bool = True,
    # --- new params ---
    initial_coords: Optional[Tensor] = None,
    ref_time_steps: int = 50,
    # --- standard folding params (0.6.1 subset used here) ---
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 200,
    num_diffn_samples: int = 5,
    seed: int | None = None,
    device: Union[str, "torch.device", None] = None,
    low_memory: bool = True,
) -> "StructureCandidates":
    """Run structure-conditioned ("truncated") diffusion refinement via Chai-1 0.6.1.

    Mirrors ``chai_lab.chai1.run_inference`` for feature preparation and output
    writing. Differs only in how the diffusion sampler is initialised (CHANGE 1/2;
    see module docstring).

    Returns ``StructureCandidates`` (same as ``run_inference`` in 0.6.1).
    chai auto-writes ``pred.model_idx_*.cif`` + ``scores.model_idx_*.npz`` to
    ``output_dir`` so ``load_prediction(output_dir)`` works.
    """
    # Defer all heavy chai imports to call time.
    import chai_lab.chai1 as chai1

    # Accept str, torch.device, or None for device (same as run_inference 0.6.1
    # caller contract; internally we work with torch.device).
    if isinstance(device, str):
        torch_device = torch.device(device)
    elif device is None:
        from xenodesign.config import resolve_device
        torch_device = torch.device(resolve_device())  # XENO_DEVICE / cuda:0 if avail / mps / cpu
    else:
        torch_device = device  # already a torch.device

    # Use 0.6.1's make_all_atom_feature_context for full feature prep (handles
    # ESM, MSA, templates, restraints, glycan bonds, etc.).
    make_all_atom_feature_context = chai1.make_all_atom_feature_context

    feature_context = make_all_atom_feature_context(
        fasta_file=fasta_file,
        output_dir=output_dir,
        use_esm_embeddings=use_esm_embeddings,
        use_msa_server=False,
        esm_device=torch_device,
    )

    candidates = run_folding_on_context_truncated(
        feature_context,
        output_dir=output_dir,
        num_trunk_recycles=num_trunk_recycles,
        num_diffn_timesteps=num_diffn_timesteps,
        num_diffn_samples=num_diffn_samples,
        seed=seed,
        device=torch_device,
        low_memory=low_memory,
        initial_coords=initial_coords,
        ref_time_steps=ref_time_steps,
    )

    return candidates


# ---------------------------------------------------------------------------
# Vendored folding function with structure-conditioned sampler init
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_folding_on_context_truncated(
    feature_context: "AllAtomFeatureContext",  # type: ignore[name-defined]
    output_dir: Path,
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 200,
    num_diffn_samples: int = 5,
    seed: int | None = None,
    device: "torch.device | None" = None,
    low_memory: bool = True,
    # --- new params ---
    initial_coords: Optional[Tensor] = None,
    ref_time_steps: int = 50,
) -> "StructureCandidates":
    """Structure-conditioned diffusion folding for chai_lab 0.6.1.

    Vendored from 0.6.1 ``run_folding_on_context``. Identical EXCEPT:
      CHANGE 1: When initial_coords is given, atom_pos is initialised from the
                prior structure perturbed by sigma[ref_idx] (not pure noise).
      CHANGE 2: The Karras update loop starts at ref_idx, running only trailing steps.

    Returns ``StructureCandidates`` (same as 0.6.1's run_folding_on_context).
    """
    import chai_lab.chai1 as chai1
    from einops import rearrange, repeat
    from tqdm import tqdm

    # --- pull all required symbols from 0.6.1 ---
    AVAILABLE_MODEL_SIZES = chai1.AVAILABLE_MODEL_SIZES
    Collate = chai1.Collate
    DiffusionConfig = chai1.DiffusionConfig
    InferenceNoiseSchedule = chai1.InferenceNoiseSchedule
    StructureCandidates = chai1.StructureCandidates
    TokenBondRestraint = chai1.TokenBondRestraint
    _bin_centers = chai1._bin_centers
    _component_moved_to = chai1._component_moved_to
    center_random_augmentation = chai1.center_random_augmentation
    feature_factory = chai1.feature_factory
    get_frames_and_mask = chai1.get_frames_and_mask
    get_scores = chai1.get_scores
    move_data_to_device = chai1.move_data_to_device
    plot_msa = chai1.plot_msa
    raise_if_msa_too_deep = chai1.raise_if_msa_too_deep
    raise_if_too_many_templates = chai1.raise_if_too_many_templates
    raise_if_too_many_tokens = chai1.raise_if_too_many_tokens
    rank = chai1.rank
    save_to_cif = chai1.save_to_cif
    set_seed = chai1.set_seed
    subsample_and_reorder_msa_feats_n_mask = chai1.subsample_and_reorder_msa_feats_n_mask
    und_self = chai1.und_self
    get_chain_letter = chai1.get_chain_letter

    # Set seed
    if seed is not None:
        set_seed([seed])

    if device is None:
        from xenodesign.config import resolve_device
        device = torch.device(resolve_device())  # XENO_DEVICE / cuda:0 if avail / mps / cpu

    # Clear memory
    torch.cuda.empty_cache()

    ##
    ## Validate inputs
    ##

    n_actual_tokens = feature_context.structure_context.num_tokens
    raise_if_too_many_tokens(n_actual_tokens)
    raise_if_too_many_templates(feature_context.template_context.num_templates)
    raise_if_msa_too_deep(feature_context.msa_context.depth)
    feature_context.structure_context.report_bonds()

    ##
    ## Prepare batch
    ##

    collator = Collate(
        feature_factory=feature_factory,
        num_key_atoms=128,
        num_query_atoms=32,
    )

    feature_contexts = [feature_context]
    batch_size = len(feature_contexts)
    batch = collator(feature_contexts)

    if not low_memory:
        batch = move_data_to_device(batch, device=device)

    # Get features and inputs from batch
    features = {name: feature for name, feature in batch["features"].items()}
    inputs = batch["inputs"]
    block_indices_h = inputs["block_atom_pair_q_idces"]
    block_indices_w = inputs["block_atom_pair_kv_idces"]
    atom_single_mask = inputs["atom_exists_mask"]
    atom_token_indices = inputs["atom_token_index"].long()
    token_single_mask = inputs["token_exists_mask"]
    token_pair_mask = und_self(token_single_mask, "b i, b j -> b i j")
    token_reference_atom_index = inputs["token_ref_atom_index"]
    atom_within_token_index = inputs["atom_within_token_index"]
    msa_mask = inputs["msa_mask"]
    template_input_masks = und_self(
        inputs["template_mask"], "b t n1, b t n2 -> b t n1 n2"
    )
    block_atom_pair_mask = inputs["block_atom_pair_mask"]

    ##
    ## Determine model size from MSA mask (0.6.1 pattern)
    ##

    _, _, model_size = msa_mask.shape
    assert model_size in AVAILABLE_MODEL_SIZES

    ##
    ## Run the features through the feature embedder (0.6.1 context-manager style)
    ##

    with _component_moved_to("feature_embedding.pt", device) as feature_embedding:
        embedded_features = feature_embedding.forward(
            crop_size=model_size,
            move_to_device=device,
            return_on_cpu=low_memory,
            **features,
        )

    token_single_input_feats = embedded_features["TOKEN"]
    token_pair_input_feats, token_pair_structure_input_feats = embedded_features[
        "TOKEN_PAIR"
    ].chunk(2, dim=-1)
    atom_single_input_feats, atom_single_structure_input_feats = embedded_features[
        "ATOM"
    ].chunk(2, dim=-1)
    block_atom_pair_input_feats, block_atom_pair_structure_input_feats = (
        embedded_features["ATOM_PAIR"].chunk(2, dim=-1)
    )
    template_input_feats = embedded_features["TEMPLATES"]
    msa_input_feats = embedded_features["MSA"]

    ##
    ## Bond feature generator (new in 0.6.1)
    ##

    bond_ft_gen = TokenBondRestraint()
    bond_ft = bond_ft_gen.generate(batch=batch).data
    with _component_moved_to("bond_loss_input_proj.pt", device) as bond_loss_input_proj:
        trunk_bond_feat, structure_bond_feat = bond_loss_input_proj.forward(
            return_on_cpu=low_memory,
            move_to_device=device,
            crop_size=model_size,
            input=bond_ft,
        ).chunk(2, dim=-1)
    token_pair_input_feats += trunk_bond_feat
    token_pair_structure_input_feats += structure_bond_feat

    ##
    ## Run the inputs through the token input embedder
    ##

    with _component_moved_to("token_embedder.pt", device) as token_input_embedder:
        token_input_embedder_outputs: tuple[Tensor, ...] = token_input_embedder.forward(
            return_on_cpu=low_memory,
            move_to_device=device,
            token_single_input_feats=token_single_input_feats,
            token_pair_input_feats=token_pair_input_feats,
            atom_single_input_feats=atom_single_input_feats,
            block_atom_pair_feat=block_atom_pair_input_feats,
            block_atom_pair_mask=block_atom_pair_mask,
            block_indices_h=block_indices_h,
            block_indices_w=block_indices_w,
            atom_single_mask=atom_single_mask,
            atom_token_indices=atom_token_indices,
            crop_size=model_size,
        )
    token_single_initial_repr, token_single_structure_input, token_pair_initial_repr = (
        token_input_embedder_outputs
    )

    ##
    ## Run the input representations through the trunk
    ##

    token_single_trunk_repr = token_single_initial_repr
    token_pair_trunk_repr = token_pair_initial_repr
    for _ in tqdm(range(num_trunk_recycles), desc="Trunk recycles"):
        subsampled_msa_input_feats, subsampled_msa_mask = None, None
        if 0 > 0:  # recycle_msa_subsample=0 → always skip subsampling
            subsampled_msa_input_feats, subsampled_msa_mask = (
                subsample_and_reorder_msa_feats_n_mask(
                    msa_input_feats,
                    msa_mask,
                )
            )
        with _component_moved_to("trunk.pt", device) as trunk:
            (token_single_trunk_repr, token_pair_trunk_repr) = trunk.forward(
                move_to_device=device,
                token_single_trunk_initial_repr=token_single_initial_repr,
                token_pair_trunk_initial_repr=token_pair_initial_repr,
                token_single_trunk_repr=token_single_trunk_repr,  # recycled
                token_pair_trunk_repr=token_pair_trunk_repr,  # recycled
                msa_input_feats=(
                    subsampled_msa_input_feats
                    if subsampled_msa_input_feats is not None
                    else msa_input_feats
                ),
                msa_mask=(
                    subsampled_msa_mask if subsampled_msa_mask is not None else msa_mask
                ),
                template_input_feats=template_input_feats,
                template_input_masks=template_input_masks,
                token_single_mask=token_single_mask,
                token_pair_mask=token_pair_mask,
                crop_size=model_size,
            )

    # In case trunk fragmented mem too much
    torch.cuda.empty_cache()

    ##
    ## Denoise with structure-conditioned init (CHANGE 1 + CHANGE 2)
    ##

    atom_single_mask = atom_single_mask.to(device)

    static_diffusion_inputs = dict(
        token_single_initial_repr=token_single_structure_input.float(),
        token_pair_initial_repr=token_pair_structure_input_feats.float(),
        token_single_trunk_repr=token_single_trunk_repr.float(),
        token_pair_trunk_repr=token_pair_trunk_repr.float(),
        atom_single_input_feats=atom_single_structure_input_feats.float(),
        atom_block_pair_input_feats=block_atom_pair_structure_input_feats.float(),
        atom_single_mask=atom_single_mask,
        atom_block_pair_mask=block_atom_pair_mask,
        token_single_mask=token_single_mask,
        block_indices_h=block_indices_h,
        block_indices_w=block_indices_w,
        atom_token_indices=atom_token_indices,
    )
    static_diffusion_inputs = move_data_to_device(
        static_diffusion_inputs, device=device
    )

    def _denoise(
        diff_mod, atom_pos: Tensor, sigma: Tensor, ds: int
    ) -> Tensor:
        atom_noised_coords = rearrange(
            atom_pos, "(b ds) ... -> b ds ...", ds=ds
        ).contiguous()
        noise_sigma = repeat(sigma, " -> b ds", b=batch_size, ds=ds)
        return diff_mod.forward(
            atom_noised_coords=atom_noised_coords.float(),
            noise_sigma=noise_sigma.float(),
            crop_size=model_size,
            **static_diffusion_inputs,
        )

    inference_noise_schedule = InferenceNoiseSchedule(
        s_max=DiffusionConfig.S_tmax,
        s_min=4e-4,
        p=7.0,
        sigma_data=DiffusionConfig.sigma_data,
    )
    sigmas = inference_noise_schedule.get_schedule(
        device=device, num_timesteps=num_diffn_timesteps
    )
    gammas = torch.where(
        (sigmas >= DiffusionConfig.S_tmin) & (sigmas <= DiffusionConfig.S_tmax),
        min(DiffusionConfig.S_churn / num_diffn_timesteps, math.sqrt(2) - 1),
        0.0,
    )

    sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

    # Compute ref_idx for truncated start
    ref_idx = max(0, num_diffn_timesteps - ref_time_steps)

    # Padded atom count (from the collated batch)
    _, num_atoms = atom_single_mask.shape

    # CHANGE 1: Structure-conditioned init vs. pure-noise init
    if initial_coords is None:
        # Original 0.6.1: pure noise from largest sigma
        atom_pos = sigmas[0] * torch.randn(
            batch_size * num_diffn_samples, num_atoms, 3, device=device
        )
    else:
        # Structure-conditioned: perturb prior by sigma[ref_idx].
        # initial_coords contains only the real atoms (n_real_atoms × 3).
        # The model's atom_pos tensor is padded to num_atoms; pad with zeros.
        # Real atoms occupy the first n_real_atoms slots; padding slots are
        # masked out by atom_single_mask so their values don't affect the model.
        coords_device = initial_coords.to(device=device, dtype=torch.float32)
        n_real = coords_device.shape[0]
        if n_real < num_atoms:
            pad = torch.zeros(
                num_atoms - n_real, 3, device=device, dtype=torch.float32
            )
            coords_padded = torch.cat([coords_device, pad], dim=0)  # (num_atoms, 3)
        else:
            if n_real > num_atoms:
                print(
                    f"[chai_truncated] WARNING: initial_coords has {n_real} atoms but the "
                    f"collated batch has {num_atoms}; truncating — the D-seed may be "
                    f"corrupted (check the CIF for extra HETATM/altloc/H atoms).",
                    flush=True,
                )
            coords_padded = coords_device[:num_atoms]
        # Expand to (batch_size * num_diffn_samples, num_atoms, 3)
        base = coords_padded.reshape(1, num_atoms, 3).expand(
            batch_size * num_diffn_samples, num_atoms, 3
        ).contiguous()
        atom_pos = base + sigmas[ref_idx] * torch.randn_like(base)

    # CHANGE 2: Start loop at ref_idx instead of 0
    loop_schedule = (
        sigmas_and_gammas[ref_idx:]
        if initial_coords is not None
        else sigmas_and_gammas
    )

    with _component_moved_to("diffusion_module.pt", device=device) as diffusion_module:
        for sigma_curr, sigma_next, gamma_curr in tqdm(
            loop_schedule, desc="Diffusion steps"
        ):
            # Center coords (chai applies random rotation+translation each step)
            atom_pos = center_random_augmentation(
                atom_pos,
                atom_single_mask=repeat(
                    atom_single_mask,
                    "b a -> (b ds) a",
                    ds=num_diffn_samples,
                ),
            )

            # Alg 2. lines 4-6
            noise = DiffusionConfig.S_noise * torch.randn(
                atom_pos.shape, device=atom_pos.device
            )
            sigma_hat = sigma_curr + gamma_curr * sigma_curr
            atom_pos_noise = (sigma_hat**2 - sigma_curr**2).clamp_min(1e-6).sqrt()
            atom_pos_hat = atom_pos + noise * atom_pos_noise

            # Lines 7-8
            denoised_pos = _denoise(
                diff_mod=diffusion_module,
                atom_pos=atom_pos_hat,
                sigma=sigma_hat,
                ds=num_diffn_samples,
            )
            d_i = (atom_pos_hat - denoised_pos) / sigma_hat
            atom_pos = atom_pos_hat + (sigma_next - sigma_hat) * d_i

            # Lines 9-11
            if sigma_next != 0 and DiffusionConfig.second_order:  # second order update
                denoised_pos = _denoise(
                    diff_mod=diffusion_module,
                    atom_pos=atom_pos,
                    sigma=sigma_next,
                    ds=num_diffn_samples,
                )
                d_i_prime = (atom_pos - denoised_pos) / sigma_next
                atom_pos = atom_pos + (sigma_next - sigma_hat) * ((d_i_prime + d_i) / 2)

    del static_diffusion_inputs
    torch.cuda.empty_cache()

    ##
    ## Run the confidence model (UNCHANGED from 0.6.1)
    ##

    with _component_moved_to("confidence_head.pt", device=device) as confidence_head:
        confidence_outputs: list[tuple[Tensor, ...]] = [
            confidence_head.forward(
                move_to_device=device,
                token_single_input_repr=token_single_initial_repr,
                token_single_trunk_repr=token_single_trunk_repr,
                token_pair_trunk_repr=token_pair_trunk_repr,
                token_single_mask=token_single_mask,
                atom_single_mask=atom_single_mask,
                atom_coords=atom_pos[ds : ds + 1],
                token_reference_atom_index=token_reference_atom_index,
                atom_token_index=atom_token_indices,
                atom_within_token_index=atom_within_token_index,
                crop_size=model_size,
            )
            for ds in range(num_diffn_samples)
        ]

    pae_logits, pde_logits, plddt_logits = [
        torch.cat(single_sample, dim=0)
        for single_sample in zip(*confidence_outputs, strict=True)
    ]

    assert atom_pos.shape[0] == num_diffn_samples
    assert pae_logits.shape[0] == num_diffn_samples

    def softmax_einsum_and_cpu(
        logits: Tensor, bin_mean: Tensor, pattern: str
    ) -> Tensor:
        from einops import einsum
        res = einsum(
            logits.float().softmax(dim=-1), bin_mean.to(logits.device), pattern
        )
        return res.to(device="cpu")

    token_mask_1d = rearrange(token_single_mask, "1 b -> b")

    pae_scores = softmax_einsum_and_cpu(
        pae_logits[:, token_mask_1d, :, :][:, :, token_mask_1d, :],
        _bin_centers(0.0, 32.0, 64),
        "b n1 n2 d, d -> b n1 n2",
    )

    pde_scores = softmax_einsum_and_cpu(
        pde_logits[:, token_mask_1d, :, :][:, :, token_mask_1d, :],
        _bin_centers(0.0, 32.0, 64),
        "b n1 n2 d, d -> b n1 n2",
    )

    plddt_scores_atom = softmax_einsum_and_cpu(
        plddt_logits,
        _bin_centers(0, 1, plddt_logits.shape[-1]),
        "b a d, d -> b a",
    )

    # converting per-atom plddt to per-token
    [mask] = atom_single_mask.cpu()
    [indices] = atom_token_indices.cpu()

    def avg_per_token_1d(x):
        n = torch.bincount(indices[mask], weights=x[mask])
        d = torch.bincount(indices[mask]).clamp(min=1)
        return n / d

    plddt_scores = torch.stack([avg_per_token_1d(x) for x in plddt_scores_atom])

    ##
    ## Write the outputs (UNCHANGED from 0.6.1)
    ##

    # Move data to the CPU so we don't hit GPU memory limits
    inputs = move_data_to_device(inputs, torch.device("cpu"))
    atom_pos = atom_pos.cpu()
    plddt_logits = plddt_logits.cpu()
    pae_logits = pae_logits.cpu()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Only plot MSA if there's actual MSA data (0.6.1 pattern)
    if feature_context.msa_context.mask.any():
        msa_plot_path = plot_msa(
            input_tokens=feature_context.structure_context.token_residue_type,
            msa_tokens=feature_context.msa_context.tokens,
            out_fname=output_dir / "msa_depth.pdf",
        )
    else:
        msa_plot_path = None

    cif_paths: list[Path] = []
    ranking_data: list = []

    for idx in range(num_diffn_samples):
        _, valid_frames_mask = get_frames_and_mask(
            atom_pos[idx : idx + 1],
            inputs["token_asym_id"],
            inputs["token_residue_index"],
            inputs["token_backbone_frame_mask"],
            inputs["token_centre_atom_index"],
            inputs["token_exists_mask"],
            inputs["atom_exists_mask"],
            inputs["token_backbone_frame_index"],
            inputs["atom_token_index"],
        )

        ranking_outputs = rank(
            atom_pos[idx : idx + 1],
            atom_mask=inputs["atom_exists_mask"],
            atom_token_index=inputs["atom_token_index"],
            token_exists_mask=inputs["token_exists_mask"],
            token_asym_id=inputs["token_asym_id"],
            token_entity_type=inputs["token_entity_type"],
            token_valid_frames_mask=valid_frames_mask,
            lddt_logits=plddt_logits[idx : idx + 1],
            lddt_bin_centers=_bin_centers(0, 1, plddt_logits.shape[-1]).to(
                plddt_logits.device
            ),
            pae_logits=pae_logits[idx : idx + 1],
            pae_bin_centers=_bin_centers(0.0, 32.0, 64).to(pae_logits.device),
        )

        ranking_data.append(ranking_outputs)

        cif_out_path = output_dir.joinpath(f"pred.model_idx_{idx}.cif")
        aggregate_score = ranking_outputs.aggregate_score.item()
        print(f"Score={aggregate_score:.4f}, writing output to {cif_out_path}")

        scaled_plddt_scores_per_atom = 100 * plddt_scores_atom[idx : idx + 1]

        # 0.6.1 uses save_to_cif with asym_entity_names dict (not outputs_to_cif)
        save_to_cif(
            coords=atom_pos[idx : idx + 1],
            bfactors=scaled_plddt_scores_per_atom,
            output_batch=inputs,
            write_path=cif_out_path,
            asym_entity_names={
                i: get_chain_letter(i)
                for i, _ in enumerate(feature_context.chains, start=1)
            },
        )
        cif_paths.append(cif_out_path)

        scores_out_path = output_dir.joinpath(f"scores.model_idx_{idx}.npz")
        np.savez(scores_out_path, **get_scores(ranking_outputs))

    return StructureCandidates(
        cif_paths=cif_paths,
        ranking_data=ranking_data,
        msa_coverage_plot_path=msa_plot_path,
        pae=pae_scores,
        pde=pde_scores,
        plddt=plddt_scores,
    )
